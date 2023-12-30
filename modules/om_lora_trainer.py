import os
import re
import toml
from time import time
#from accelerate import Accelerator
import modules.om_logging as oml
from safetensors.torch import load_file as torch_load_safetensors
from torch import load as torch_load_ckpt
import modules.om_train_network as om_train_network
import modules.om_observer as omo

class OMLoRATrainer():
  def __init__(s,observer:omo.OMObserver):
    s.observer=observer
    return
  
  def configure_globals(s):
    # These carry information from past executions
    #if "model_url" in globals(): old_model_url = model_url
    #else: old_model_url = None
    if "dependencies_installed" not in globals(): dependencies_installed = False
    #s.model_file=None
    s.custom_dataset = None
    s.override_dataset_config_file = None
    s.override_config_file=None
    s.optimizer = "AdamW8bit"
    if "optimizer_args" not in globals(): optimizer_args = None
    s.continue_from_lora = ""
    s.weighted_captions = False
    s.adjust_tags = False
    s.keep_tokens_weight = 1.0
    return

  def raw_setup(s):
    s.COLAB = False #True # low ram
    s.XFORMERS = True
    COMMIT = "9a67e0df390033a89f17e70df5131393692c2a55"
    BETTER_EPOCH_NAMES = True
    LOAD_TRUNCATED_IMAGES = True
    #folder_structure = "Organize by project (MyDrive/Loras/project_name/dataset)" #@param ["Organize by category (MyDrive/lora_training/datasets/project_name)", "Organize by project (MyDrive/Loras/project_name/dataset)"]
    s.custom_model_is_based_on_sd2 = False #@param {type:"boolean"}
    #@markdown Resolution of 512 is standard for Stable Diffusion 1.5. Higher resolution training is much slower but can lead to better details. <p>
    #@markdown Images will be automatically scaled while training to produce the best results, so you don't need to crop or resize anything yourself.
    s.resolution = 512 #@param {type:"slider", min:512, max:1024, step:128}
    #@markdown This option will train your images both normally and flipped, for no extra cost, to learn more from them. Turn it on specially if you have less than 20 images. <p>
    s.flip_aug = False #@param {type:"boolean"} markdown **Turn it off if you care about asymmetrical elements in your Lora**.
    s.caption_extension = ".txt" #param {type:"string"}     #markdown Leave empty for no captions.
    shuffle_tags = True #@param {type:"boolean"} #@markdown Shuffling anime tags in place improves learning and prompting. An activation tag goes at the start of every text file and will not be shuffled.
    s.shuffle_caption = shuffle_tags
    activation_tags = "1" #@param [0,1,2,3]
    s.keep_tokens = int(activation_tags)
    #@markdown ### ▶️ Steps <p>
    #@markdown Your images will repeat this number of times during training. I recommend that your images multiplied by their repeats is between 200 and 400.
    #s.num_image_repeats = 10 #@param {type:"number"}
    #@markdown Choose how long you want to train for. A good starting point is around 10 epochs or around 2000 steps.<p>
    #@markdown One epoch is a number of steps equal to: your number of images multiplied by their repeats, divided by batch size. <p>
    preferred_unit = "Epochs" #@param ["Epochs", "Steps"]
    #how_many = 10 #@param {type:"number"}
    #max_train_epochs = #how_many if preferred_unit == "Epochs" else None
    #max_train_steps = how_many if preferred_unit == "Steps" else None
    #@markdown Saving more epochs will let you compare your Lora's progress better.
    s.save_every_n_epochs = 1 #@param {type:"number"}
    s.keep_only_last_n_epochs = 10 #@param {type:"number"}
    if not s.save_every_n_epochs: s.save_every_n_epochs = s.max_epochs
    if not s.keep_only_last_n_epochs: s.keep_only_last_n_epochs = s.max_epochs
    #@markdown Increasing the batch size makes training faster, but may make learning worse. Recommended 2 or 3.    
    #@markdown ### ▶️ Learning
    #@markdown The learning rate is the most important for your results. If you want to train slower with lots of images, or if your dim and alpha are high, move the unet to 2e-4 or lower. <p>
    #@markdown The text encoder helps your Lora learn concepts slightly better. It is recommended to make it half or a fifth of the unet. If you're training a style you can even set it to 0.
    s.unet_lr = 5e-4 #@param {type:"number"}
    s.text_encoder_lr = 1e-4 #@param {type:"number"}
    #@markdown The scheduler is the algorithm that guides the learning rate. If you're not sure, pick `constant` and ignore the number. I personally recommend `cosine_with_restarts` with 3 restarts.
    s.lr_scheduler = "cosine_with_restarts" #@param ["constant", "cosine", "cosine_with_restarts", "constant_with_warmup", "linear", "polynomial"]
    lr_scheduler_number = 3 #@param {type:"number"}
    s.lr_scheduler_num_cycles = lr_scheduler_number if s.lr_scheduler == "cosine_with_restarts" else 0
    lr_scheduler_power = lr_scheduler_number if s.lr_scheduler == "polynomial" else 0
    #@markdown Steps spent "warming up" the learning rate during training for efficiency. I recommend leaving it at 5%.
    lr_warmup_ratio = 0.05 #@param {type:"slider", min:0.0, max:0.5, step:0.01}
    s.lr_warmup_steps = 0
    #@markdown New feature that adjusts loss over time, makes learning much more efficient, and training can be done with about half as many epochs. Uses a value of 5.0 as recommended by [the paper](https://arxiv.org/abs/2303.09556).
    min_snr_gamma = True #@param {type:"boolean"}
    s.min_snr_gamma_value = 5.0 if min_snr_gamma else None
    #@markdown ### ▶️ Structure
    #@markdown LoRA is the classic type and good for a variety of purposes. LoCon is good with artstyles as it has more layers to learn more aspects of the dataset.
    lora_type = "LoRA" #@param ["LoRA", "LoCon"]
    #@markdown Below are some recommended values for the following settings:
    #@markdown | type | network_dim | network_alpha | conv_dim | conv_alpha |
    #@markdown | :---: | :---: | :---: | :---: | :---: |
    #@markdown | LoRA | 16 | 8 |   |   |
    #@markdown | LoCon | 16 | 8 | 8 | 4 |
    #@markdown More dim means larger Lora, it can hold more information but more isn't always better. A dim between 8-32 is recommended, and alpha equal to half the dim.
    s.network_dim = 16 #@param {type:"slider", min:1, max:128, step:1}
    s.network_alpha = 8 #@param {type:"slider", min:1, max:128, step:1}
    #@markdown The following two values only apply to the additional layers of LoCon.
    conv_dim = 8 #@param {type:"slider", min:1, max:64, step:1}
    conv_alpha = 4 #@param {type:"slider", min:1, max:64, step:1}
    s.network_module = "modules.networks.lora" #"networks.lora"
    s.network_args = None
    s.optimizer_args={} # Newly added
    if lora_type.lower() == "locon": network_args = [f"conv_dim={conv_dim}", f"conv_alpha={conv_alpha}"]
    #@markdown ### ▶️ Ready
    #@markdown You can now run this cell to cook your Lora. Good luck! <p>
    # 👩‍💻 Cool code goes here
    if s.optimizer.lower() == "prodigy" or "dadapt" in s.optimizer.lower():
      #if override_values_for_dadapt_and_prodigy:
        #unet_lr = 0.5
        #text_encoder_lr = 0.5
        #lr_scheduler = "constant_with_warmup"
        #lr_warmup_ratio = 0.05
        #network_alpha = network_dim
      if not s.optimizer_args:
        s.optimizer_args = ["decouple=True","weight_decay=0.01","betas=[0.9,0.999]"]
        if s.optimizer == "Prodigy":
          s.optimizer_args.extend(["d_coef=2","use_bias_correction=True"])
          if lr_warmup_ratio > 0:
            s.optimizer_args.append("safeguard_warmup=True")
          else:
            s.optimizer_args.append("safeguard_warmup=False")
    s.deps_dir=os.path.join(s.root_dir, "deps")
    s.repo_dir=os.path.join(s.root_dir, "kohya-trainer")
    s.images_folder=os.path.join(s.root_dir, "datasets")
    s.output_folder=os.path.join(s.root_dir, "output")
    s.config_folder=os.path.join(s.root_dir, "config")
    s.log_folder=os.path.join(s.root_dir, "log")
    s.config_file=os.path.join(s.config_folder, "training_config.toml")
    s.dataset_config_file = os.path.join(s.config_folder, "dataset_config.toml")
    s.accelerate_config_file = os.path.join(s.repo_dir, "accelerate_config/config.yaml")

  def install_dependencies(s):
    from accelerate.utils import write_basic_config
    if not os.path.exists(accelerate_config_file):
      write_basic_config(save_location=accelerate_config_file)
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    os.environ["BITSANDBYTES_NOWELCOME"] = "1"
    os.environ["SAFETENSORS_FAST_GPU"] = "1"

  def validate_dataset(s):
    #global lr_warmup_steps, lr_warmup_ratio, caption_extension, keep_tokens, keep_tokens_weight, weighted_captions, adjust_tags
    supported_types = (".png", ".jpg", ".jpeg", ".webp", ".bmp")
    oml.debug("\n💿 Checking dataset...")
    if not s.project_name.strip() or any(c in s.project_name for c in " .()\"'\\/"):
      oml.error("Please choose a valid project name.")
      return

    #if custom_dataset:
      #oml.debug("using custom dataset")
      #try:
      #  datconf = toml.loads(custom_dataset)
      #  datasets = [d for d in datconf["datasets"][0]["subsets"]]
      #except:
      #  oml.error(f"Your custom dataset is invalid or contains an error! Please check the original template.")
      #  return
      #reg = [d.get("image_dir") for d in datasets if d.get("is_reg", False)]
      #datasets_dict = {d["image_dir"]: d["num_repeats"] for d in datasets}
      #folders = datasets_dict.keys()
      #files = [f for folder in folders for f in os.listdir(folder)]
      #images_repeats = {folder: (len([f for f in os.listdir(folder) if f.lower().endswith(supported_types)]), datasets_dict[folder]) for folder in folders}
    else:
      oml.debug("using other dataset")
      reg = []
      folders = [s.images_folder]
      oml.debug(f"images_folder={s.images_folder}")
      files = os.listdir(s.images_folder)
      #files = os.listdir("~/Loras/camilla_martin_lora_121220231002/dataset")
      #files = os.listdir("/home/ubuntu/Loras/camilla_martin_lora_121220231002/dataset")
      oml.debug(f"files={files}")
      #for f in files: oml.debug(f"file={f.lower()}")
      images_repeats = {s.images_folder: (len([f for f in files if f.lower().endswith(supported_types)]), s.num_image_repeats)}

    oml.debug(f"images_repeats={images_repeats}")
    for folder in folders:
      if not os.path.exists(folder):
        raise Exception(f"💥 Error: The folder {folder.replace('/content/drive/', '')} doesn't exist.")
        return
    for folder, (img, rep) in images_repeats.items():
      oml.debug(f"img={img}")
      oml.debug(f"folder={folder}")
      if not img: raise Exception("Make sure you put image files such as .png or .jpg in the folder and not a .zip! Also don't use ~ as it does not seem to always work!")
    for f in files:
      if not f.lower().endswith(".txt") and not f.lower().endswith(supported_types):
        raise Exception(f"💥 Error: Invalid file in dataset: \"{f}\". Aborting.")

    if not [txt for txt in files if txt.lower().endswith(".txt")]:
      caption_extension = ""
    if s.continue_from_lora and not (s.continue_from_lora.endswith(".safetensors") and os.path.exists(s.continue_from_lora)):
      raise Exception(f"💥 Error: Invalid path to existing Lora. Example: /content/drive/MyDrive/Loras/example.safetensors")

    pre_steps_per_epoch = sum(img*rep for (img, rep) in images_repeats.values())
    steps_per_epoch = pre_steps_per_epoch/s.batch_size
    total_steps = int(s.max_epochs*steps_per_epoch)
    estimated_epochs = int(total_steps/steps_per_epoch)
    s.observer.observe(s.observer.TRANING_START_VALIDATE_EVENT, args=(pre_steps_per_epoch,steps_per_epoch,total_steps,estimated_epochs))
    #lr_warmup_steps = int(total_steps*lr_warmup_ratio)

    for folder, (img, rep) in images_repeats.items():
      print("📁 "+folder.replace("/content/drive/", "") + (" (Regularization)" if folder in reg else ""))
      oml.success(f"Found {img} images")
      print(f"📈 {img} images with {rep} repeats, equaling {img*rep} steps.")
    print(f"📉 Divide {pre_steps_per_epoch} steps by {s.batch_size} batch size to get {steps_per_epoch} steps per epoch.")
    #if s.smax_train_epochs: print(f"🔮 There will be {s.max_train_epochs} epochs, for around {total_steps} total training steps.")
    #else:
    print(f"🔮 There will be {total_steps} steps, divided into {estimated_epochs} epochs and then some.")

    if total_steps > 10000:
      raise Exception("💥 Error: Your total steps are too high. You probably made a mistake. Aborting...")
      return

    if s.adjust_tags:
      print(f"\n📎 Weighted tags: {'ON' if s.weighted_captions else 'OFF'}")
      if s.weighted_captions:
        print(f"📎 Will use {s.keep_tokens_weight} weight on {s.keep_tokens} activation tag(s)")
      print("📎 Adjusting tags...")
      s.adjust_weighted_tags(folders, s.keep_tokens, s.keep_tokens_weight, s.weighted_captions)

    return True

  def adjust_weighted_tags(s,folders, keep_tokens: int, keep_tokens_weight: float, weighted_captions: bool):
    weighted_tag = re.compile(r"\((.+?):[.\d]+\)(,|$)")
    for folder in folders:
      for txt in [f for f in os.listdir(folder) if f.lower().endswith(".txt")]:
        with open(os.path.join(folder, txt), 'r') as f:
          content = f.read()
        # reset previous changes
        content = content.replace('\\', '')
        content = weighted_tag.sub(r'\1\2', content)
        if weighted_captions:
          # re-apply changes
          content = content.replace(r'(', r'\(').replace(r')', r'\)').replace(r':', r'\:')
          if keep_tokens_weight > 1:
            tags = [s.strip() for s in content.split(",")]
            for i in range(min(keep_tokens, len(tags))):
              tags[i] = f'({tags[i]}:{keep_tokens_weight})'
            content = ", ".join(tags)
        with open(os.path.join(folder, txt), 'w') as f:
          f.write(content)

  def create_config(s):
    if s.override_config_file:
      config_file = s.override_config_file
      oml.warn(f"⭕ Using custom config file {config_file}")
    else:
      oml.progress("Creating new config dict!")
      config_dict = {
        "additional_network_arguments": {
          "unet_lr": s.unet_lr,
          "text_encoder_lr": s.text_encoder_lr,
          "network_dim": s.network_dim,
          "network_alpha": s.network_alpha,
          "network_module": s.network_module,
          "network_args": s.network_args,
          "network_train_unet_only": True if s.text_encoder_lr == 0 else None,
          "network_weights": s.continue_from_lora if s.continue_from_lora else None
        },
        "optimizer_arguments": {
          "learning_rate": s.unet_lr,
          "lr_scheduler": s.lr_scheduler,
          "lr_scheduler_num_cycles": s.lr_scheduler_num_cycles if s.lr_scheduler == "cosine_with_restarts" else None,
          "lr_scheduler_power": s.lr_scheduler_power if s.lr_scheduler == "polynomial" else None,
          "lr_warmup_steps": s.lr_warmup_steps if s.lr_scheduler != "constant" else None,
          "optimizer_type": s.optimizer,
          "optimizer_args": s.optimizer_args if s.optimizer_args else None,
        },
        "training_arguments": {
          #"max_train_steps": s.max_train_steps,
          "max_train_epochs": s.max_epochs, #renamed from max_train_epochs
          "save_every_n_epochs": s.save_every_n_epochs,
          "save_last_n_epochs": s.keep_only_last_n_epochs,
          "train_batch_size": s.batch_size,
          "noise_offset": None,
          "clip_skip": 2,
          "min_snr_gamma": s.min_snr_gamma_value,
          "weighted_captions": s.weighted_captions,
          "seed": 42,
          "max_token_length": 225,
          "xformers": s.XFORMERS,
          "lowram": s.COLAB,
          "max_data_loader_n_workers": 8,
          "persistent_data_loader_workers": True,
          "save_precision": "fp16",
          "mixed_precision": "fp16",
          "output_dir": s.output_folder,
          "logging_dir": s.log_folder,
          "output_name": s.project_name,
          "log_prefix": s.project_name,
        },
        "model_arguments": {
          "pretrained_model_name_or_path": s.model_file,
          "v2": s.custom_model_is_based_on_sd2,
          "v_parameterization": True if s.custom_model_is_based_on_sd2 else None,
        },
        "saving_arguments": {
          "save_model_as": "safetensors",
        },
        "dreambooth_arguments": {
          "prior_loss_weight": 1.0,
        },
        "dataset_arguments": {
          "cache_latents": True,
        },
      }

      for key in config_dict:
        if isinstance(config_dict[key], dict):
          config_dict[key] = {k: v for k, v in config_dict[key].items() if v is not None}

      with open(s.config_file, "w") as f:
        f.write(toml.dumps(config_dict))
      print(f"\n📄 Config saved to {s.config_file}")

    if s.override_dataset_config_file:
      dataset_config_file = s.override_dataset_config_file
      print(f"⭕ Using custom dataset config file {dataset_config_file}")
    else:
      dataset_config_dict = {
        "general": {
          "resolution": s.resolution,
          "shuffle_caption": s.shuffle_caption,
          "keep_tokens": s.keep_tokens,
          "flip_aug": s.flip_aug,
          "caption_extension": s.caption_extension,
          "enable_bucket": True,
          "bucket_reso_steps": 64,
          "bucket_no_upscale": False,
          "min_bucket_reso": 320 if s.resolution > 640 else 256,
          "max_bucket_reso": 1280 if s.resolution > 640 else 1024,
        },
        "datasets": toml.loads(s.custom_dataset)["datasets"] if s.custom_dataset else [
          {
            "subsets": [
              {
                "num_repeats": s.num_image_repeats, # Newly renamed
                "image_dir": s.images_folder,
                "class_tokens": None if s.caption_extension else s.project_name
              }
            ]
          }
        ]
      }

      for key in dataset_config_dict:
        if isinstance(dataset_config_dict[key], dict):
          dataset_config_dict[key] = {k: v for k, v in dataset_config_dict[key].items() if v is not None}

      with open(s.dataset_config_file, "w") as f:
        f.write(toml.dumps(dataset_config_dict))
      print(f"📄 Dataset config saved to {s.dataset_config_file}")

  def strip_filename(s,file_path):
      head, tail = os.path.split(file_path)
      return head

  def download_model(s):
    oml.debug(f"model_file={s.model_file}")
    model_file_path=f"{s.model_cache_folder}/{s.model_file}"
    oml.debug(f"Model file path={model_file_path}")
    
    from urllib import request
    if os.path.exists(model_file_path):
      oml.debug(f"Already downloaded {s.model_file}, skipping new download!")
    else:
      raise Exception(f"Download of models currently not supported, please download the model {s.model_file} manually and place in the folder {s.model_cache_folder}!")
      #os.makedirs(s.model_cache_folder, exist_ok=True)
      #oml.debug(f"Trying to downoad model via real_model_url={real_model_url} to local file {model_file}...")  
      #httpResp=request.urlretrieve(remote_url, local_file) # Skip if exists  
      #oml.debug(f"httpResp={httpResp}")

    if s.model_file.lower().endswith(".safetensors"):      
      try:
        test = torch_load_safetensors(model_file_path)
        del test
      except Exception as e:
        raise Exception("Ever called this?")
        #if "HeaderTooLarge" in str(e):
        new_model_file = os.path.splitext(model_file_path)[0]+".ckpt"
        #!mv "{model_file}" "{new_model_file}"
        model_file = new_model_file
        print(f"Renamed model to {os.path.splitext(model_file_path)[0]}.ckpt")

    if s.model_file.lower().endswith(".ckpt"):      
      raise Exception("Ever called this 2?")
      try:
        test = torch_load_ckpt(s.model_file)
        del test
      except Exception as e:
        return False

    return True

  def download_model_org(s):
    raise Exception("Is this method ever called???")
    global old_model_url, model_url, model_file
    real_model_url = model_url.strip()  

    if real_model_url.lower().endswith((".ckpt", ".safetensors")):
      model_file = f"{s.root_dir}/models{real_model_url[real_model_url.rfind('/'):]}"
      oml.debug(f"Model file={model_file}")
    else:
      model_file = f"{s.root_dir}/models/downloaded_model.safetensors"
      
    oml.debug(f"download_model() - step1")
    if m := re.search(r"(?:https?://)?(?:www\.)?huggingface\.co/[^/]+/[^/]+/blob", model_url):
      real_model_url = real_model_url.replace("blob", "resolve")
    elif m := re.search(r"(?:https?://)?(?:www\\.)?civitai\.com/models/([0-9]+)(/[A-Za-z0-9-_]+)?", model_url):
      if m.group(2):
        model_file = f"/content{m.group(2)}.safetensors"
      if m := re.search(r"modelVersionId=([0-9]+)", model_url):
        real_model_url = f"https://civitai.com/api/download/models/{m.group(1)}"
      else:
        raise Valueoml.error("optional_custom_training_model_url contains a civitai link, but the link doesn't include a modelVersionId. You can also right click the download button to copy the direct download link.")

  #  !aria2c "{real_model_url}" --console-log-level=warn -c -s 16 -x 16 -k 10M -d / -o "{model_file}"
    oml.debug(f"download_model() - step2")
    from urllib import request
    remote_url = real_model_url
    local_file = model_file
    model_folder=strip_filename(local_file)
    oml.debug(f"model_folder={model_folder}")
    if os.path.exists(model_file):
      oml.debug(f"Already downloaded {model_file}, skipping new download!")
    else:
      os.makedirs(model_folder, exist_ok=True)
      oml.debug(f"Trying to downoad model via real_model_url={real_model_url} to local file {model_file}...")  
      httpResp=request.urlretrieve(remote_url, local_file) # Skip if exists  
      oml.debug(f"httpResp={httpResp}")

    if model_file.lower().endswith(".safetensors"):
      from safetensors.torch import load_file as load_safetensors
      try:
        test = load_safetensors(model_file)
        del test
      except Exception as e:
        #if "HeaderTooLarge" in str(e):
        new_model_file = os.path.splitext(model_file)[0]+".ckpt"
        #!mv "{model_file}" "{new_model_file}"
        model_file = new_model_file
        print(f"Renamed model to {os.path.splitext(model_file)[0]}.ckpt")

    if model_file.lower().endswith(".ckpt"):
      from torch import load as load_ckpt
      try:
        test = load_ckpt(model_file)
        del test
      except Exception as e:
        return False

    return True

  def create_folder_structure(s):
    oml.info(f"Root folder is [{s.root_dir}]")
    for dir in (s.root_dir, s.deps_dir, s.repo_dir, s.log_folder, s.images_folder, s.output_folder, s.config_folder):
      oml.debug(f"Creating folder structure [{dir}]")
      os.makedirs(dir, exist_ok=True)
    oml.success(f"Folder structure created.")

  #def start_training(s,project_name:str,project_dir:str,batch_size:int,max_epochs:int,model_file:str,model_cache_folder:str,num_image_repeats:int=10):
  def start_training(s,settings,hyper_parameters):
    s.project_name=settings.project_name
    s.project_dir=settings.project_dir
    s.batch_size=hyper_parameters.batch_size
    s.max_epochs=hyper_parameters.max_epochs
    s.model_file=settings.model_file
    s.num_image_repeats=hyper_parameters.num_image_repeats
    s.model_cache_folder=settings.model_cache_folder #model_cache_folder=f"{s.root_dir}/model_cache"
    s.root_dir=os.path.join(s.project_dir,s.project_name)
    if(s.project_name==None or s.project_name==''): raise Exception("Project name must be defined")
    if(s.project_name==None or s.project_dir==''): raise Exception("Project dir must be defined")
    
    s.configure_globals()
    s.raw_setup()
    oml.success(f"Starting LoRA project [{settings.project_name}]")
    s.create_folder_structure()
    if not s.validate_dataset(): return
    #if old_model_url != model_url or not model_file or not os.path.exists(model_file):
    s.download_model()
    s.create_config()
    print("\n⭐ Starting trainer...\n")
    #os.chdir(s.repo_dir)
    # https://huggingface.co/docs/accelerate/basic_tutorials/launch    
    om_train_network.train_network_main(observer=s.observer,settings=settings,hyper_parameters=hyper_parameters,dataset_config_file=s.dataset_config_file,config_file=s.config_file,accelerate_config_file=s.accelerate_config_file,num_cpu_threads_per_process=1)
    oml.success("You can now download your Lora!")
    #if not get_ipython().__dict__['user_ns']['_exit_code']:
    #display(Markdown("### ✅ Done! [Go download your Lora from Google Drive](https://drive.google.com/drive/my-drive)\n"
    #                   "### There will be several files, you should try the latest version (the file with the largest number next to it)"))

