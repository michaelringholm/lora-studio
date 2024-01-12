import os
import random
import sys
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import numpy as np
import torch
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from modules.captions.blip.blip import blip_decoder
#import library.train_util as train_util
import modules.om_ext_train_util as train_util
import modules.om_logging as oml

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMAGE_SIZE = 384
IMAGE_TRANSFORM = transforms.Compose(
    [
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE), interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ]
)

class ImageLoadingTransformDataset(torch.utils.data.Dataset):
    def __init__(self, image_paths):
        self.images = image_paths

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]

        try:
            image = Image.open(img_path).convert("RGB")
            # convert to tensor temporarily so dataloader will accept it
            tensor = IMAGE_TRANSFORM(image)
        except Exception as e:
            print(f"Could not load image path / 画像を読み込めません: {img_path}, error: {e}")
            return None

        return (tensor, img_path)

def collate_fn_remove_corrupted(batch):
    """Collate function that allows to remove corrupted examples in the
    dataloader. It expects that the dataloader returns 'None' when that occurs.
    The 'None's in the batch are removed.
    """
    # Filter out all the Nones (corrupted examples)
    batch = list(filter(lambda x: x is not None, batch))
    return batch

def make_captions(project_folder:str,project_name:str):
    oml.debug("make_captions() called...")
    settings=type('', (), {})() 
    settings.seed=42
    settings.train_data_dir=os.path.join(project_folder,project_name,"datasets")
    settings.recursive=False
    settings.caption_weights="https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_large_caption.pth"
    settings.max_data_loader_n_workers=None
    settings.batch_size=1
    settings.beam_search=None
    settings.top_p=None
    settings.caption_extension=".txt" #type=str, default=".caption", help="extension of caption file
    settings.num_beams=1 #type=int, default=1, help="num of beams in beam search /beam search
    settings.top_p:float=0.9 #type=float, default=0.9, help="top_p in Nucleus sampling / Nucleus sampling
    settings.max_length:int=75 #type=int, default=75, help="max length of caption / caption
    settings.min_length:int=5 #type=int, default=5, help="min length of caption / caption
    settings.debug=False
    # fix the seed for reproducibility
    seed = settings.seed  # + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    #if not os.path.exists("blip"):
        #args.train_data_dir = os.path.abspath(args.train_data_dir)  # convert to absolute path
        #cwd = os.getcwd()
        #print("Current Working Directory is: ", cwd)
        #os.chdir("finetune")

    oml.debug(f"load images from {settings.train_data_dir}")
    train_data_dir_path = Path(settings.train_data_dir)
    image_paths = train_util.glob_images_pathlib(train_data_dir_path, settings.recursive)
    oml.debug(f"found {len(image_paths)} images.")

    oml.debug(f"loading BLIP caption: {settings.caption_weights}")
    model = blip_decoder(pretrained=settings.caption_weights, image_size=IMAGE_SIZE, vit="large", med_config="./modules/captions/blip/med_config.json")
    model.eval()
    model = model.to(DEVICE)
    oml.debug("BLIP loaded")

    # captioning
    def run_batch(path_imgs):
        imgs = torch.stack([im for _, im in path_imgs]).to(DEVICE)

        with torch.no_grad():
            if settings.beam_search:
                captions = model.generate(imgs, sample=False, num_beams=settings.num_beams, max_length=settings.max_length, min_length=settings.min_length)
            else:
                captions = model.generate(imgs, sample=True, top_p=settings.top_p, max_length=settings.max_length, min_length=settings.min_length)

        for (image_path, _), caption in zip(path_imgs, captions):
            with open(os.path.splitext(image_path)[0] + settings.caption_extension, "wt", encoding="utf-8") as f:
                f.write(caption + "\n")
                if settings.debug:
                    print(image_path, caption)

    # 読み込みの高速化のためにDataLoaderを使うオプション
    if settings.max_data_loader_n_workers is not None:
        dataset = ImageLoadingTransformDataset(image_paths)
        data = torch.utils.data.DataLoader(
            dataset,
            batch_size=settings.batch_size,
            shuffle=False,
            num_workers=settings.max_data_loader_n_workers,
            collate_fn=collate_fn_remove_corrupted,
            drop_last=False,
        )
    else:
        data = [[(None, ip)] for ip in image_paths]

    b_imgs = []
    for data_entry in tqdm(data, smoothing=0.0):
        for data in data_entry:
            if data is None:
                continue

            img_tensor, image_path = data
            if img_tensor is None:
                try:
                    raw_image = Image.open(image_path)
                    if raw_image.mode != "RGB":
                        raw_image = raw_image.convert("RGB")
                    img_tensor = IMAGE_TRANSFORM(raw_image)
                except Exception as e:
                    oml.error(f"Could not load image path / 画像を読み込めません: {image_path}, error: {e}")
                    continue

            b_imgs.append((image_path, img_tensor))
            if len(b_imgs) >= settings.batch_size:
                run_batch(b_imgs)
                b_imgs.clear()
    if len(b_imgs) > 0:
        run_batch(b_imgs)
    return



