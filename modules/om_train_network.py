# region imports
import importlib
import argparse
import gc
import math
import os
import sys
import random
import time
import json
from multiprocessing import Value
import toml
import modules.om_logging as oml
from tqdm import tqdm
import torch
import GPUtil    
try:
    import intel_extension_for_pytorch as ipex

    if torch.xpu.is_available():
        from modules.ipex import ipex_init

        ipex_init()
except Exception:
    pass
from accelerate.utils import set_seed
from diffusers import DDPMScheduler
import modules.om_ext_model_util as model_util

import modules.om_ext_train_util as om_ext_train_util
from modules.om_ext_train_util import (
    DreamBoothDataset,
)
import modules.om_ext_config_util as config_util
from modules.om_ext_config_util import (
    ConfigSanitizer,
    BlueprintGenerator,
)
import modules.om_ext_huggingface_util as huggingface_util
import modules.om_ext_custom_train_functions as custom_train_functions
from modules.om_ext_custom_train_functions import (
    apply_snr_weight,
    get_weighted_text_embeddings,
    prepare_scheduler_for_custom_training,
    scale_v_prediction_loss_like_noise_prediction,
    add_v_prediction_like_loss,
    apply_debiased_estimation,
)
import modules.om_observer as omo
import modules.om_hyper_params as omhp
import modules.om_general_settings as omgs
# endregion

class NetworkTrainer:
    def __init__(self,observer:omo.OMObserver):
        self.vae_scale_factor = 0.18215
        self.is_sdxl = False
        self.observer=observer

    # TODO Generalize with other scripts.
    def generate_step_logs(self, args: argparse.Namespace, current_loss, avr_loss, lr_scheduler, keys_scaled=None, mean_norm=None, maximum_norm=None):
        logs = {"loss/current": current_loss, "loss/average": avr_loss}
        if keys_scaled is not None:
            logs["max_norm/keys_scaled"] = keys_scaled
            logs["max_norm/average_key_norm"] = mean_norm
            logs["max_norm/max_key_norm"] = maximum_norm

        lrs = lr_scheduler.get_last_lr()
        if args.network_train_text_encoder_only or len(lrs) <= 2:  # not block lr (or single block)
            if args.network_train_unet_only:
                logs["lr/unet"] = float(lrs[0])
            elif args.network_train_text_encoder_only:
                logs["lr/textencoder"] = float(lrs[0])
            else:
                logs["lr/textencoder"] = float(lrs[0])
                logs["lr/unet"] = float(lrs[-1])  # may be same to textencoder

            if (args.optimizer_type.lower().startswith("DAdapt".lower()) or args.optimizer_type.lower() == "Prodigy".lower()):  # tracking d*lr value of unet.
                logs["lr/d*lr"] = (lr_scheduler.optimizers[-1].param_groups[0]["d"] * lr_scheduler.optimizers[-1].param_groups[0]["lr"])
        else:
            idx = 0
            if not args.network_train_unet_only:
                logs["lr/textencoder"] = float(lrs[0])
                idx = 1

            for i in range(idx, len(lrs)):
                logs[f"lr/group{i}"] = float(lrs[i])
                if args.optimizer_type.lower().startswith("DAdapt".lower()) or args.optimizer_type.lower() == "Prodigy".lower():
                    logs[f"lr/d*lr/group{i}"] = ( lr_scheduler.optimizers[-1].param_groups[i]["d"] * lr_scheduler.optimizers[-1].param_groups[i]["lr"]  )
        return logs

    def assert_extra_args(self, args, train_dataset_group):
        pass

    def load_target_model(self, args, weight_dtype, accelerator):
        print(f"args.pretrained_model_name_or_path={args.pretrained_model_name_or_path}")
        text_encoder, vae, unet, _ = om_ext_train_util.load_target_model(args, weight_dtype, accelerator)
        return model_util.get_model_version_str_for_sd1_sd2(args.v2, args.v_parameterization), text_encoder, vae, unet

    def load_tokenizer(self, args):
        tokenizer = om_ext_train_util.load_tokenizer(args)
        return tokenizer

    def is_text_encoder_outputs_cached(self, args):
        return False

    def is_train_text_encoder(self, args):
        return not args.network_train_unet_only and not self.is_text_encoder_outputs_cached(args)

    def cache_text_encoder_outputs_if_needed(
        self, args, accelerator, unet, vae, tokenizers, text_encoders, data_loader, weight_dtype
    ):
        for t_enc in text_encoders:
            t_enc.to(accelerator.device)

    def get_text_cond(self, args, accelerator, batch, tokenizers, text_encoders, weight_dtype):
        input_ids = batch["input_ids"].to(accelerator.device)
        encoder_hidden_states = om_ext_train_util.get_hidden_states(args, input_ids, tokenizers[0], text_encoders[0], weight_dtype)
        return encoder_hidden_states

    def call_unet(self, args, accelerator, unet, noisy_latents, timesteps, text_conds, batch, weight_dtype):
        noise_pred = unet(noisy_latents, timesteps, text_conds).sample
        return noise_pred

    def sample_images(self, accelerator, args, epoch, global_step, device, vae, tokenizer, text_encoder, unet):
        om_ext_train_util.sample_images(accelerator, args, epoch, global_step, device, vae, tokenizer, text_encoder, unet)

    def load_user_custom_config(s,args):
        oml.debug(f"load_user_custom_config() -> Loading dataset config from {args.dataset_config}")
        user_config = config_util.load_user_config(args.dataset_config)
        ignored = ["train_data_dir", "reg_data_dir", "in_json"]
        if any(getattr(args, attr) is not None for attr in ignored):
            oml.warn("ignoring the following options because config file is found: {0}".format(", ".join(ignored)))
        return user_config
    
    def load_dreambooth_config(s,args):
        oml.info("load_dreambooth_config() -> Using DreamBooth method.")
        user_config = {
            "datasets": [
                {
                    "subsets": config_util.generate_dreambooth_subsets_config_by_subdirs(
                        args.train_data_dir, args.reg_data_dir
                    )
                }
            ]
        }
        return user_config
    
    def build_config(s,args):
        oml.debug("build_config()")
        user_config = {
            "datasets": [
                {
                    "subsets": [
                        {
                            "image_dir": args.train_data_dir,
                            "metadata_file": args.in_json,
                        }
                    ]
                }
            ]
        }
        return user_config
    
    def get_user_config(s,args):
        use_dreambooth_method = args.in_json is None
        use_user_config = args.dataset_config is not None
        user_config=None
        if use_user_config:
            user_config=s.load_user_custom_config(args)
        else:
            if use_dreambooth_method:
                user_config=s.load_dreambooth_config(args)
            else:
                user_config=s.build_config()
        return user_config,use_dreambooth_method,use_user_config
    
    def build_metadata(s,args,total_batch_size,dataset,dataset_dirs_info,reg_dataset_dirs_info):
        metadata={
            "ss_batch_size_per_device": args.train_batch_size,
            "ss_total_batch_size": total_batch_size,
            "ss_resolution": args.resolution,
            "ss_color_aug": bool(args.color_aug),
            "ss_flip_aug": bool(args.flip_aug),
            "ss_random_crop": bool(args.random_crop),
            "ss_shuffle_caption": bool(args.shuffle_caption),
            "ss_enable_bucket": bool(dataset.enable_bucket),
            "ss_bucket_no_upscale": bool(dataset.bucket_no_upscale),
            "ss_min_bucket_reso": dataset.min_bucket_reso,
            "ss_max_bucket_reso": dataset.max_bucket_reso,
            "ss_keep_tokens": args.keep_tokens,
            "ss_dataset_dirs": json.dumps(dataset_dirs_info),
            "ss_reg_dataset_dirs": json.dumps(reg_dataset_dirs_info),
            "ss_tag_frequency": json.dumps(dataset.tag_frequency),
            "ss_bucket_info": json.dumps(dataset.bucket_info),
        }
        return metadata
    
    def build_more_metadata(s,args,session_id,training_started_at,train_dataset_group,train_dataloader,num_train_epochs,model_version,optimizer_name,optimizer_args):
        metadata={
            "ss_session_id": session_id,  # random integer indicating which group of epochs the model came from
            "ss_training_started_at": training_started_at,  # unix timestamp
            "ss_output_name": args.output_name,
            "ss_learning_rate": args.learning_rate,
            "ss_text_encoder_lr": args.text_encoder_lr,
            "ss_unet_lr": args.unet_lr,
            "ss_num_train_images": train_dataset_group.num_train_images,
            "ss_num_reg_images": train_dataset_group.num_reg_images,
            "ss_num_batches_per_epoch": len(train_dataloader),
            "ss_num_epochs": num_train_epochs,
            "ss_gradient_checkpointing": args.gradient_checkpointing,
            "ss_gradient_accumulation_steps": args.gradient_accumulation_steps,
            "ss_max_train_steps": args.max_train_steps,
            "ss_lr_warmup_steps": args.lr_warmup_steps,
            "ss_lr_scheduler": args.lr_scheduler,
            "ss_network_module": args.network_module,
            "ss_network_dim": args.network_dim,  # None means default because another network than LoRA may have another default dim
            "ss_network_alpha": args.network_alpha,  # some networks may not have alpha
            "ss_network_dropout": args.network_dropout,  # some networks may not have dropout
            "ss_mixed_precision": args.mixed_precision,
            "ss_full_fp16": bool(args.full_fp16),
            "ss_v2": bool(args.v2),
            "ss_base_model_version": model_version,
            "ss_clip_skip": args.clip_skip,
            "ss_max_token_length": args.max_token_length,
            "ss_cache_latents": bool(args.cache_latents),
            "ss_seed": args.seed,
            "ss_lowram": args.lowram,
            "ss_noise_offset": args.noise_offset,
            "ss_multires_noise_iterations": args.multires_noise_iterations,
            "ss_multires_noise_discount": args.multires_noise_discount,
            "ss_adaptive_noise_scale": args.adaptive_noise_scale,
            "ss_zero_terminal_snr": args.zero_terminal_snr,
            "ss_training_comment": args.training_comment,  # will not be updated after training
            "ss_sd_scripts_commit_hash": om_ext_train_util.get_git_revision_hash(),
            "ss_optimizer": optimizer_name + (f"({optimizer_args})" if len(optimizer_args) > 0 else ""),
            "ss_max_grad_norm": args.max_grad_norm,
            "ss_caption_dropout_rate": args.caption_dropout_rate,
            "ss_caption_dropout_every_n_epochs": args.caption_dropout_every_n_epochs,
            "ss_caption_tag_dropout_rate": args.caption_tag_dropout_rate,
            "ss_face_crop_aug_range": args.face_crop_aug_range,
            "ss_prior_loss_weight": args.prior_loss_weight,
            "ss_min_snr_gamma": args.min_snr_gamma,
            "ss_scale_weight_norms": args.scale_weight_norms,
            "ss_ip_noise_gamma": args.ip_noise_gamma,
            "ss_debiased_estimation": bool(args.debiased_estimation_loss),
        }
        return metadata
    
    def print_accelerator(s,accelerator,train_dataset_group,train_dataloader,num_train_epochs,args):
        accelerator.print("running training")
        accelerator.print(f"  num train images * repeats: {train_dataset_group.num_train_images}")
        accelerator.print(f"  num reg images: {train_dataset_group.num_reg_images}")
        accelerator.print(f"  num batches per epoch / 1epoch: {len(train_dataloader)}")
        accelerator.print(f"  num epochs / epoch: {num_train_epochs}")
        accelerator.print(f"  batch size per device: {', '.join([str(d.batch_size) for d in train_dataset_group.datasets])}")
        # accelerator.print(f"  total train batch size (with parallel & distributed & accumulation) ）: {total_batch_size}")
        accelerator.print(f"  gradient accumulation steps = {args.gradient_accumulation_steps}")
        accelerator.print(f"  total optimization steps: {args.max_train_steps}")
    
    def train(self, args):
        oml.debug("train() - step1")
        session_id = random.randint(0, 2**32)
        training_started_at = time.time()
        om_ext_train_util.verify_training_args(args)
        om_ext_train_util.prepare_dataset_args(args, True)
        cache_latents = args.cache_latents
        if args.seed is None: args.seed = random.randint(0, 2**32)
        set_seed(args.seed)
        tokenizer = self.load_tokenizer(args)
        tokenizers = tokenizer if isinstance(tokenizer, list) else [tokenizer]

        if args.dataset_class is None:
            blueprint_generator = BlueprintGenerator(ConfigSanitizer(True, True, False, True),observer=self.observer)
            user_config,use_dreambooth_method,use_user_config=self.get_user_config(args)            
            blueprint = blueprint_generator.generate(user_config, args, tokenizer=tokenizer)
            print(f"train_network.blueprint.dataset_group={blueprint.dataset_group}")
            train_dataset_group = config_util.generate_dataset_group_by_blueprint(blueprint.dataset_group,self.observer)
        else: train_dataset_group = om_ext_train_util.load_arbitrary_dataset(args, tokenizer) # use arbitrary dataset class

        current_epoch = Value("i", 0)
        current_step = Value("i", 0)
        ds_for_collator = train_dataset_group if args.max_data_loader_n_workers == 0 else None
        collator = om_ext_train_util.collator_class(current_epoch, current_step, ds_for_collator)
        if args.debug_dataset:
            om_ext_train_util.debug_dataset(train_dataset_group)
            return
        if len(train_dataset_group) == 0:
            print("No data found. Please verify arguments (train_data_dir must be the parent of folders with images)")
            return
        if cache_latents: assert (train_dataset_group.is_latent_cacheable()), "when caching latents, either color_aug or random_crop cannot be used / latentをキャッシュするときはcolor_augとrandom_cropは使えません"
        self.assert_extra_args(args, train_dataset_group)
        print("preparing accelerator")
        oml.debug("train() - preparing accelerator...")
        accelerator = om_ext_train_util.prepare_accelerator(args)
        is_main_process = accelerator.is_main_process

        # mixed precisionに対応した型を用意しておき適宜castする
        weight_dtype, save_dtype = om_ext_train_util.prepare_dtype(args)
        vae_dtype = torch.float32 if args.no_half_vae else weight_dtype
        oml.debug("train() - loading target model...")
        model_version, text_encoder, vae, unet = self.load_target_model(args, weight_dtype, accelerator)
        # text_encoder is List[CLIPTextModel] or CLIPTextModel
        text_encoders = text_encoder if isinstance(text_encoder, list) else [text_encoder]
        om_ext_train_util.replace_unet_modules(unet, args.mem_eff_attn, args.xformers, args.sdpa)
        if torch.__version__ >= "2.0.0":  vae.set_use_memory_efficient_attention_xformers(args.xformers)
        # 差分追加学習のためにモデルを読み込む
        sys.path.append(os.path.dirname(__file__))
        accelerator.print("import network module:", args.network_module)
        oml.debug("train() - importing network module...")
        network_module = importlib.import_module(args.network_module)

        if args.base_weights is not None:
            # base_weights が指定されている場合は、指定された重みを読み込みマージする
            for i, weight_path in enumerate(args.base_weights):
                if args.base_weights_multiplier is None or len(args.base_weights_multiplier) <= i:
                    multiplier = 1.0
                else:
                    multiplier = args.base_weights_multiplier[i]
                accelerator.print(f"merging module: {weight_path} with multiplier {multiplier}")
                module, weights_sd = network_module.create_network_from_weights(multiplier, weight_path, vae, text_encoder, unet, for_inference=True)
                module.merge_to(text_encoder, unet, weights_sd, weight_dtype, accelerator.device if args.lowram else "cpu")
            accelerator.print(f"all weights merged: {', '.join(args.base_weights)}")

        # 学習を準備する
        if cache_latents:
            vae.to(accelerator.device, dtype=vae_dtype)
            vae.requires_grad_(False)
            vae.eval()
            with torch.no_grad(): train_dataset_group.cache_latents(vae, args.vae_batch_size, args.cache_latents_to_disk, accelerator.is_main_process)
            vae.to("cpu")
            if torch.cuda.is_available(): torch.cuda.empty_cache()
            gc.collect()
            accelerator.wait_for_everyone()

        # Text Encoder cpu/gpu
        self.cache_text_encoder_outputs_if_needed(args, accelerator, unet, vae, tokenizers, text_encoders, train_dataset_group, weight_dtype)
        # prepare network
        net_kwargs = {}
        if args.network_args is not None:
            for net_arg in args.network_args:
                key, value = net_arg.split("=")
                net_kwargs[key] = value

        # if a new network is added in future, add if ~ then blocks for each network (;'∀')
        if args.dim_from_weights:
            network, _ = network_module.create_network_from_weights(1, args.network_weights, vae, text_encoder, unet, **net_kwargs)
        else:
            if "dropout" not in net_kwargs:                
                net_kwargs["dropout"] = args.network_dropout # workaround for LyCORIS
            network = network_module.create_network(1.0,args.network_dim,args.network_alpha,vae,text_encoder,unet,neuron_dropout=args.network_dropout,**net_kwargs,)
        if network is None: return

        if hasattr(network, "prepare_network"):
            network.prepare_network(args)
        if args.scale_weight_norms and not hasattr(network, "apply_max_norm_regularization"):
            print("warning: scale_weight_norms is specified but the network does not support it / scale_weight_normsが指定されていますが、ネットワークが対応していません")
            args.scale_weight_norms = False

        train_unet = not args.network_train_text_encoder_only
        train_text_encoder = self.is_train_text_encoder(args)
        network.apply_to(text_encoder, unet, train_text_encoder, train_unet)

        if args.network_weights is not None:
            acc_info = network.load_weights(args.network_weights)
            accelerator.print(f"load network weights from {args.network_weights}: {acc_info}")

        if args.gradient_checkpointing:
            unet.enable_gradient_checkpointing()
            for t_enc in text_encoders:
                t_enc.gradient_checkpointing_enable()
            del t_enc
            network.enable_gradient_checkpointing()  # may have no effect

        accelerator.print("prepare optimizer, data loader etc.")

        try:
            trainable_params = network.prepare_optimizer_params(args.text_encoder_lr, args.unet_lr, args.learning_rate)
        except TypeError:
            accelerator.print("Deprecated: use prepare_optimizer_params(text_encoder_lr, unet_lr, learning_rate) instead of prepare_optimizer_params(text_encoder_lr, unet_lr)")
            trainable_params = network.prepare_optimizer_params(args.text_encoder_lr, args.unet_lr)

        optimizer_name, optimizer_args, optimizer = om_ext_train_util.get_optimizer(args, trainable_params)

        # DataLoader
        n_workers = min(args.max_data_loader_n_workers, os.cpu_count() - 1)  # cpu_count-1
        oml.debug(f"planning to use {n_workers} workers...")
        if n_workers > 1: raise Exception("This wont work on Windows!")

        train_dataloader = torch.utils.data.DataLoader(
            train_dataset_group,
            batch_size=1,
            shuffle=True,
            collate_fn=collator,
            num_workers=n_workers,
            persistent_workers=args.persistent_data_loader_workers,
        )

        if args.max_train_epochs is not None:
            args.max_train_steps = args.max_train_epochs * math.ceil(len(train_dataloader) / accelerator.num_processes / args.gradient_accumulation_steps)
            accelerator.print(f"override steps. steps for {args.max_train_epochs} epochs is / 指定エポックまでのステップ数: {args.max_train_steps}")

        train_dataset_group.set_max_train_steps(args.max_train_steps)
        # lr scheduler
        lr_scheduler = om_ext_train_util.get_scheduler_fix(args, optimizer, accelerator.num_processes)
        # Experimental Feature: Perform training with fp16/bf16, including gradients. Convert the entire model to fp16/bf16.
        if args.full_fp16:
            assert (args.mixed_precision == "fp16"), "full_fp16 requires mixed precision='fp16' / full_fp16を使う場合はmixed_precision='fp16'を指定してください。"
            accelerator.print("enable full fp16 training.")
            network.to(weight_dtype)
        elif args.full_bf16:
            assert (args.mixed_precision == "bf16"), "full_bf16 requires mixed precision='bf16' / full_bf16を使う場合はmixed_precision='bf16'を指定してください。"
            accelerator.print("enable full bf16 training.")
            network.to(weight_dtype)

        unet.requires_grad_(False)
        unet.to(dtype=weight_dtype)
        for t_enc in text_encoders:t_enc.requires_grad_(False)
        # It seems like the accelerator is doing something useful.
        # TODO The code is excessively redundant, so let's organize it.
        if train_unet and train_text_encoder:
            if len(text_encoders) > 1:
                unet, t_enc1, t_enc2, network, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(unet, text_encoders[0], text_encoders[1], network, optimizer, train_dataloader, lr_scheduler)
                text_encoder = text_encoders = [t_enc1, t_enc2]
                del t_enc1, t_enc2
            else:
                unet, text_encoder, network, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(unet, text_encoder, network, optimizer, train_dataloader, lr_scheduler)
                text_encoders = [text_encoder]
        elif train_unet:
            unet, network, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(unet, network, optimizer, train_dataloader, lr_scheduler)
            for t_enc in text_encoders: t_enc.to(accelerator.device, dtype=weight_dtype)
        elif train_text_encoder:
            if len(text_encoders) > 1:
                t_enc1, t_enc2, network, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(text_encoders[0], text_encoders[1], network, optimizer, train_dataloader, lr_scheduler)
                text_encoder = text_encoders = [t_enc1, t_enc2]
                del t_enc1, t_enc2
            else:
                text_encoder, network, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(text_encoder, network, optimizer, train_dataloader, lr_scheduler)
                text_encoders = [text_encoder]

            unet.to(accelerator.device, dtype=weight_dtype)  # move to device because unet is not prepared by accelerator
        else:
            network, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(network, optimizer, train_dataloader, lr_scheduler)

        # transform DDP after prepare (train_network here only)
        oml.debug("train() - step3")
        text_encoders = om_ext_train_util.transform_models_if_DDP(text_encoders)
        unet, network = om_ext_train_util.transform_models_if_DDP([unet, network])

        if args.gradient_checkpointing:
            # according to TI example in Diffusers, train is required
            unet.train()
            for t_enc in text_encoders:
                t_enc.train()

                # set top parameter requires_grad = True for gradient checkpointing works
                if train_text_encoder:
                    t_enc.text_model.embeddings.requires_grad_(True)

            # set top parameter requires_grad = True for gradient checkpointing works
            if not train_text_encoder:  # train U-Net only
                unet.parameters().__next__().requires_grad_(True)
        else:
            unet.eval()
            for t_enc in text_encoders:
                t_enc.eval()

        del t_enc

        network.prepare_grad_etc(text_encoder, unet)

        if not cache_latents:  # Prepare VAE as VAE is used when caching is not performed.
            vae.requires_grad_(False)
            vae.eval()
            vae.to(accelerator.device, dtype=vae_dtype)

        # Experimental feature: Perform fp16 training, including gradients. Apply a patch to PyTorch to enable grad scaling in fp16.
        if args.full_fp16:
            om_ext_train_util.patch_accelerator_for_fp16_training(accelerator)

        # resume
        om_ext_train_util.resume_from_local_or_hf_if_specified(accelerator, args)

        # epoch Calculate the number.
        num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
        num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)
        if (args.save_n_epoch_ratio is not None) and (args.save_n_epoch_ratio > 0):
            args.save_every_n_epochs = math.floor(num_train_epochs / args.save_n_epoch_ratio) or 1

        # Train
        # TODO: find a way to handle total batch size when there are multiple datasets
        total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
        self.print_accelerator(accelerator,train_dataset_group,train_dataloader,num_train_epochs,args)
        # TODO refactor metadata creation and move to util
        oml.debug("train() - step4")
        metadata=self.build_more_metadata(args,session_id,training_started_at,train_dataset_group,train_dataloader,num_train_epochs,model_version,optimizer_name,optimizer_args)
        if use_user_config:
            # save metadata of multiple datasets
            # NOTE: pack "ss_datasets" value as json one time or should also pack nested collections as json?
            datasets_metadata = []
            tag_frequency = {}  # merge tag frequency for metadata editor
            dataset_dirs_info = {}  # merge subset dirs for metadata editor
            oml.debug("train() - step6")
            for dataset in train_dataset_group.datasets:
                is_dreambooth_dataset = isinstance(dataset, DreamBoothDataset)
                dataset_metadata = {
                    "is_dreambooth": is_dreambooth_dataset,
                    "batch_size_per_device": dataset.batch_size,
                    "num_train_images": dataset.num_train_images,  # includes repeating
                    "num_reg_images": dataset.num_reg_images,
                    "resolution": (dataset.width, dataset.height),
                    "enable_bucket": bool(dataset.enable_bucket),
                    "min_bucket_reso": dataset.min_bucket_reso,
                    "max_bucket_reso": dataset.max_bucket_reso,
                    "tag_frequency": dataset.tag_frequency,
                    "bucket_info": dataset.bucket_info,
                }

                subsets_metadata = []
                for subset in dataset.subsets:
                    subset_metadata = {
                        "img_count": subset.img_count,
                        "num_repeats": subset.num_repeats,
                        "color_aug": bool(subset.color_aug),
                        "flip_aug": bool(subset.flip_aug),
                        "random_crop": bool(subset.random_crop),
                        "shuffle_caption": bool(subset.shuffle_caption),
                        "keep_tokens": subset.keep_tokens,
                    }

                    image_dir_or_metadata_file = None
                    if subset.image_dir:
                        image_dir = os.path.basename(subset.image_dir)
                        subset_metadata["image_dir"] = image_dir
                        image_dir_or_metadata_file = image_dir

                    if is_dreambooth_dataset:
                        subset_metadata["class_tokens"] = subset.class_tokens
                        subset_metadata["is_reg"] = subset.is_reg
                        if subset.is_reg:
                            image_dir_or_metadata_file = None  # not merging reg dataset
                    else:
                        metadata_file = os.path.basename(subset.metadata_file)
                        subset_metadata["metadata_file"] = metadata_file
                        image_dir_or_metadata_file = metadata_file  # may overwrite

                    subsets_metadata.append(subset_metadata)

                    # merge dataset dir: not reg subset only
                    # TODO update additional-network extension to show detailed dataset config from metadata
                    if image_dir_or_metadata_file is not None:
                        # datasets may have a certain dir multiple times
                        v = image_dir_or_metadata_file
                        i = 2
                        while v in dataset_dirs_info:
                            v = image_dir_or_metadata_file + f" ({i})"
                            i += 1
                        image_dir_or_metadata_file = v

                        dataset_dirs_info[image_dir_or_metadata_file] = {
                            "n_repeats": subset.num_repeats,
                            "img_count": subset.img_count,
                        }

                dataset_metadata["subsets"] = subsets_metadata
                datasets_metadata.append(dataset_metadata)

                # merge tag frequency:
                for ds_dir_name, ds_freq_for_dir in dataset.tag_frequency.items():
                    # If a directory is used by multiple datasets, count it only once.
                    # Originally, the number of repetitions is specified, so the occurrence count of tags within captions and how many times it is used in training do not necessarily match.
                    # Therefore, aggregating the counts for multiple datasets here may not make much sense.
                    if ds_dir_name in tag_frequency:
                        continue
                    tag_frequency[ds_dir_name] = ds_freq_for_dir

            metadata["ss_datasets"] = json.dumps(datasets_metadata)
            metadata["ss_tag_frequency"] = json.dumps(tag_frequency)
            metadata["ss_dataset_dirs"] = json.dumps(dataset_dirs_info)
        else:
            # conserving backward compatibility when using train_dataset_dir and reg_dataset_dir
            assert (
                len(train_dataset_group.datasets) == 1
            ), f"There should be a single dataset but {len(train_dataset_group.datasets)} found. This seems to be a bug. / データセットは1個だけ存在するはずですが、実際には{len(train_dataset_group.datasets)}個でした。プログラムのバグかもしれません。"

            dataset = train_dataset_group.datasets[0]

            dataset_dirs_info = {}
            reg_dataset_dirs_info = {}
            if use_dreambooth_method:
                for subset in dataset.subsets:
                    acc_info = reg_dataset_dirs_info if subset.is_reg else dataset_dirs_info
                    acc_info[os.path.basename(subset.image_dir)] = {"n_repeats": subset.num_repeats, "img_count": subset.img_count}
            else:
                for subset in dataset.subsets:
                    dataset_dirs_info[os.path.basename(subset.metadata_file)] = {
                        "n_repeats": subset.num_repeats,
                        "img_count": subset.img_count,
                    }

            metadata.update(self.build_metadata(args,total_batch_size,dataset,dataset_dirs_info,reg_dataset_dirs_info))

        # add extra args
        if args.network_args: metadata["ss_network_args"] = json.dumps(net_kwargs)

        # model name and hash
        oml.debug("train() - step8")
        if args.pretrained_model_name_or_path is not None:
            sd_model_name = args.pretrained_model_name_or_path
            if os.path.exists(sd_model_name):
                metadata["ss_sd_model_hash"] = om_ext_train_util.model_hash(sd_model_name)
                metadata["ss_new_sd_model_hash"] = om_ext_train_util.calculate_sha256(sd_model_name)
                sd_model_name = os.path.basename(sd_model_name)
            metadata["ss_sd_model_name"] = sd_model_name

        if args.vae is not None:
            vae_name = args.vae
            if os.path.exists(vae_name):
                metadata["ss_vae_hash"] = om_ext_train_util.model_hash(vae_name)
                metadata["ss_new_vae_hash"] = om_ext_train_util.calculate_sha256(vae_name)
                vae_name = os.path.basename(vae_name)
            metadata["ss_vae_name"] = vae_name

        metadata = {k: str(v) for k, v in metadata.items()}

        # make minimum metadata for filtering
        minimum_metadata = {}
        for key in om_ext_train_util.SS_METADATA_MINIMUM_KEYS:
            if key in metadata:
                minimum_metadata[key] = metadata[key]

        # TODO Observer
        progress_bar = tqdm(range(args.max_train_steps), smoothing=0, disable=not accelerator.is_local_main_process, desc="steps")
        global_step = 0
        noise_scheduler = DDPMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000, clip_sample=False)
        prepare_scheduler_for_custom_training(noise_scheduler, accelerator.device)
        if args.zero_terminal_snr:
            custom_train_functions.fix_noise_scheduler_betas_for_zero_terminal_snr(noise_scheduler)

        if accelerator.is_main_process:
            init_kwargs = {}
            if args.log_tracker_config is not None:
                init_kwargs = toml.load(args.log_tracker_config)
            accelerator.init_trackers("network_train" if args.log_tracker_name is None else args.log_tracker_name, init_kwargs=init_kwargs)

        # TODO Observer
        loss_recorder = om_ext_train_util.LossRecorder()
        del train_dataset_group

        # callback for step start
        if hasattr(network, "on_step_start"): on_step_start = network.on_step_start
        else: on_step_start = lambda *args, **kwargs: None

        # function for saving/removing
        def save_model(ckpt_name, unwrapped_nw, steps, epoch_no, force_sync_upload=False):
            oml.debug(f"save_model()")
            os.makedirs(args.output_dir, exist_ok=True)
            ckpt_file = os.path.join(args.output_dir, ckpt_name)

            accelerator.print(f"\nsaving checkpoint: {ckpt_file}")
            metadata["ss_training_finished_at"] = str(time.time())
            metadata["ss_steps"] = str(steps)
            metadata["ss_epoch"] = str(epoch_no)

            metadata_to_save = minimum_metadata if args.no_metadata else metadata
            oml.debug(f"save_model() - metadata to save are {metadata_to_save}...")
            sai_metadata = om_ext_train_util.get_sai_model_spec(None, args, self.is_sdxl, True, False)
            oml.debug(f"save_model() - sai_metadata are {sai_metadata}...")
            metadata_to_save.update(sai_metadata)

            oml.debug(f"save_model() - saving weights...")
            #import modules.networks.lora_diffusers as lodi            
            #if(unwrapped_nw is lodi.LoRANetwork):
                #lora_network:lodi.LoRANetwork=network
                #lora_network.save

            unwrapped_nw.save_weights(ckpt_file, save_dtype, metadata_to_save)
            if args.huggingface_repo_id is not None:
                huggingface_util.upload(args, ckpt_file, "/" + ckpt_name, force_sync_upload=force_sync_upload)

        def remove_model(old_ckpt_name):
            old_ckpt_file = os.path.join(args.output_dir, old_ckpt_name)
            if os.path.exists(old_ckpt_file):
                accelerator.print(f"removing old checkpoint: {old_ckpt_file}")
                os.remove(old_ckpt_file)

        # training loop
        oml.debug("train() - step9")
        for epoch in range(num_train_epochs):
            accelerator.print(f"\nepoch {epoch+1}/{num_train_epochs}")
            current_epoch.value = epoch + 1
            metadata["ss_epoch"] = str(epoch + 1)
            network.on_epoch_start(text_encoder, unet)

            for step, batch in enumerate(train_dataloader):
                current_step.value = global_step
                with accelerator.accumulate(network):
                    on_step_start(text_encoder, unet)

                    with torch.no_grad():
                        if "latents" in batch and batch["latents"] is not None:
                            latents = batch["latents"].to(accelerator.device)
                            #oml.debug(f"latents_a={latents}")
                            #oml.debug(f"batch[images]={batch['images']}")
                        else:
                            latents = vae.encode(batch["images"].to(dtype=vae_dtype)).latent_dist.sample()
                            #oml.debug(f"latents_b={latents}")
                            if torch.any(torch.isnan(latents)):
                                accelerator.print("NaN found in latents, replacing with zeros")
                                latents = torch.where(torch.isnan(latents), torch.zeros_like(latents), latents)
                        latents = latents * self.vae_scale_factor
                    b_size = latents.shape[0] # TODO remove?

                    with torch.set_grad_enabled(train_text_encoder), accelerator.autocast():
                        # Get the text embedding for conditioning
                        if args.weighted_captions:
                            text_encoder_conds = get_weighted_text_embeddings(
                                tokenizer,
                                text_encoder,
                                batch["captions"],
                                accelerator.device,
                                args.max_token_length // 75 if args.max_token_length else 1,
                                clip_skip=args.clip_skip,
                            )
                        else: text_encoder_conds = self.get_text_cond(args, accelerator, batch, tokenizers, text_encoders, weight_dtype)

                    # Sample noise, sample a random timestep for each image, and add noise to the latents,
                    # with noise offset and/or multires noise if specified
                    noise, noisy_latents, timesteps = om_ext_train_util.get_noise_noisy_latents_and_timesteps(args, noise_scheduler, latents)
                    # Predict the noise residual
                    with accelerator.autocast():
                        noise_pred = self.call_unet(args, accelerator, unet, noisy_latents, timesteps, text_encoder_conds, batch, weight_dtype)

                    if args.v_parameterization: target = noise_scheduler.get_velocity(latents, noise, timesteps) # v-parameterization training
                    else: target = noise

                    loss:torch.Tensor = torch.nn.functional.mse_loss(noise_pred.float(), target.float(), reduction="none")
                    #oml.debug(f"torch.loss.mean([1, 2, 3])={loss.mean([1, 2, 3])}")
                    #oml.debug(f"torch.noise_pred={noise_pred}")
                    #oml.debug(f"torch.noise[0]={noise[0]}")
                    #oml.debug(f"torch.noisy_latents[0]={noisy_latents[0]}")
                    #oml.debug(f"torch.target[0]={target[0]}")
                    loss = loss.mean([1, 2, 3])

                    loss_weights = batch["loss_weights"]  # 各sampleごとのweight
                    loss = loss * loss_weights

                    if args.min_snr_gamma:
                        loss = apply_snr_weight(loss, timesteps, noise_scheduler, args.min_snr_gamma, args.v_parameterization)
                    if args.scale_v_pred_loss_like_noise_pred:
                        loss = scale_v_prediction_loss_like_noise_prediction(loss, timesteps, noise_scheduler)
                    if args.v_pred_like_loss:
                        loss = add_v_prediction_like_loss(loss, timesteps, noise_scheduler, args.v_pred_like_loss)
                    if args.debiased_estimation_loss:
                        loss = apply_debiased_estimation(loss, timesteps, noise_scheduler)

                    #oml.debug(f"loss before mean()={loss}")
                    loss = loss.mean()  # It's an average, so there's no need to divide by the batch size.
                    #oml.debug("train() - step9a")
                    accelerator.backward(loss)
                    if accelerator.sync_gradients and args.max_grad_norm != 0.0:
                        params_to_clip = network.get_trainable_params()
                        accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)

                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad(set_to_none=True)

                if args.scale_weight_norms:
                    keys_scaled, mean_norm, maximum_norm = network.apply_max_norm_regularization(
                        args.scale_weight_norms, accelerator.device
                    )
                    max_mean_logs = {"Keys Scaled": keys_scaled, "Average key norm": mean_norm}
                else:
                    keys_scaled, mean_norm, maximum_norm = None, None, None

                # Checks if the accelerator has performed an optimization step behind the scenes
                if accelerator.sync_gradients:
                    progress_bar.update(1)
                    global_step += 1

                    self.sample_images(accelerator, args, None, global_step, accelerator.device, vae, tokenizer, text_encoder, unet)

                    # Save the model every specified step.
                    if args.save_every_n_steps is not None and global_step % args.save_every_n_steps == 0:
                        accelerator.wait_for_everyone()
                        if accelerator.is_main_process:
                            ckpt_name = om_ext_train_util.get_step_ckpt_name(args, "." + args.save_model_as, global_step)
                            save_model(ckpt_name, accelerator.unwrap_model(network), global_step, epoch)

                            if args.save_state:
                                om_ext_train_util.save_and_remove_state_stepwise(args, accelerator, global_step)

                            remove_step_no = om_ext_train_util.get_remove_step_no(args, global_step)
                            if remove_step_no is not None:
                                remove_ckpt_name = om_ext_train_util.get_step_ckpt_name(args, "." + args.save_model_as, remove_step_no)
                                remove_model(remove_ckpt_name)

                current_loss = loss.detach().item()
                loss_recorder.add(epoch=epoch, step=step, loss=current_loss)
                avr_loss: float = loss_recorder.moving_average
                logs = {"avr_loss": avr_loss}  # , "lr": lr_scheduler.get_last_lr()[0]}
                progress_bar.set_postfix(**logs)
                oml.debug(f"current_epoch={current_epoch.value},current_loss={current_loss},avr_loss={avr_loss},global_step={global_step},loss_obj=={loss}")
                if(math.isnan(current_loss)): raise Exception(f"Current loss is NaN but {type(current_loss)}, loss={loss}")
                self.observer.observe(self.observer.TRAINING_STEP_EVENT, args=(current_epoch,step,current_loss,avr_loss,global_step))
                if args.scale_weight_norms:
                    progress_bar.set_postfix(**{**max_mean_logs, **logs})

                if args.logging_dir is not None:
                    logs = self.generate_step_logs(args, current_loss, avr_loss, lr_scheduler, keys_scaled, mean_norm, maximum_norm)
                    accelerator.log(logs, step=global_step)

                if global_step >= args.max_train_steps:
                    break

            if args.logging_dir is not None:
                logs = {"loss/epoch": loss_recorder.moving_average}
                accelerator.log(logs, step=epoch + 1)

            accelerator.wait_for_everyone()

            # Save the model every specified epoch.
            oml.debug("train() - step9c")
            if args.save_every_n_epochs is not None:
                saving = (epoch + 1) % args.save_every_n_epochs == 0 and (epoch + 1) < num_train_epochs
                if is_main_process and saving:
                    ckpt_name = om_ext_train_util.get_epoch_ckpt_name(args, "." + args.save_model_as, epoch + 1)
                    save_model(ckpt_name, accelerator.unwrap_model(network), global_step, epoch + 1)

                    remove_epoch_no = om_ext_train_util.get_remove_epoch_no(args, epoch + 1)
                    if remove_epoch_no is not None:
                        remove_ckpt_name = om_ext_train_util.get_epoch_ckpt_name(args, "." + args.save_model_as, remove_epoch_no)
                        remove_model(remove_ckpt_name)

                    if args.save_state:
                        om_ext_train_util.save_and_remove_state_on_epoch_end(args, accelerator, epoch + 1)

            self.sample_images(accelerator, args, epoch + 1, global_step, accelerator.device, vae, tokenizer, text_encoder, unet)

            # end of epoch

        # metadata["ss_epoch"] = str(num_train_epochs)
        metadata["ss_training_finished_at"] = str(time.time())

        if is_main_process:
            network = accelerator.unwrap_model(network)
            #oml.debug(f"train().unwrapped model")
            #import modules.networks.lora_diffusers as lodi
            #lora_network:lodi.LoRANetwork=network

        accelerator.end_training()

        if is_main_process and args.save_state:
            om_ext_train_util.save_state_on_train_end(args, accelerator)

        if is_main_process:
            ckpt_name = om_ext_train_util.get_last_ckpt_name(args, "." + args.save_model_as)
            oml.debug("train() -> final save of model and meta data...")
            save_model(ckpt_name, network, global_step, num_train_epochs, force_sync_upload=True)
            oml.info("model saved.")


# region global methods
def setup_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    om_ext_train_util.add_sd_models_arguments(parser)
    om_ext_train_util.add_dataset_arguments(parser, True, True, True)
    om_ext_train_util.add_training_arguments(parser, True)
    om_ext_train_util.add_optimizer_arguments(parser)
    config_util.add_config_arguments(parser)
    custom_train_functions.add_custom_train_arguments(parser)
    parser.add_argument("--no_metadata", action="store_true", help="do not save metadata in output model / メタデータを出力先モデルに保存しない")
    parser.add_argument(
        "--save_model_as",
        type=str,
        default="safetensors",
        choices=[None, "ckpt", "pt", "safetensors"],
        help="format to save the model (default is .safetensors) / モデル保存時の形式（デフォルトはsafetensors）",
    )
    parser.add_argument("--unet_lr", type=float, default=None, help="learning rate for U-Net / U-Netの学習率")
    parser.add_argument("--text_encoder_lr", type=float, default=None, help="learning rate for Text Encoder / Text Encoderの学習率")
    parser.add_argument("--network_weights", type=str, default=None, help="pretrained weights for network / 学習するネットワークの初期重み")
    parser.add_argument("--network_module", type=str, default=None, help="network module to train / 学習対象のネットワークのモジュール")
    parser.add_argument("--network_dim", type=int, default=None, help="network dimensions (depends on each network) / モジュールの次元数（ネットワークにより定義は異なります）")
    parser.add_argument(
        "--network_alpha",
        type=float,
        default=1,
        help="alpha for LoRA weight scaling, default 1 (same as network_dim for same behavior as old version) / LoRaの重み調整のalpha値、デフォルト1（旧バージョンと同じ動作をするにはnetwork_dimと同じ値を指定）",
    )
    parser.add_argument(
        "--network_dropout",
        type=float,
        default=None,
        help="Drops neurons out of training every step (0 or None is default behavior (no dropout), 1 would drop all neurons) / 訓練時に毎ステップでニューロンをdropする（0またはNoneはdropoutなし、1は全ニューロンをdropout）",
    )
    parser.add_argument(
        "--network_args", type=str, default=None, nargs="*", help="additional arguments for network (key=value) / ネットワークへの追加の引数"
    )
    parser.add_argument("--network_train_unet_only", action="store_true", help="only training U-Net part / U-Net関連部分のみ学習する")
    parser.add_argument(
        "--network_train_text_encoder_only", action="store_true", help="only training Text Encoder part / Text Encoder関連部分のみ学習する"
    )
    parser.add_argument(
        "--training_comment", type=str, default=None, help="arbitrary comment string stored in metadata / メタデータに記録する任意のコメント文字列"
    )
    parser.add_argument(
        "--dim_from_weights",
        action="store_true",
        help="automatically determine dim (rank) from network_weights / dim (rank)をnetwork_weightsで指定した重みから自動で決定する",
    )
    parser.add_argument(
        "--scale_weight_norms",
        type=float,
        default=None,
        help="Scale the weight of each key pair to help prevent overtraing via exploding gradients. (1 is a good starting point) / 重みの値をスケーリングして勾配爆発を防ぐ（1が初期値としては適当）",
    )
    parser.add_argument(
        "--base_weights",
        type=str,
        default=None,
        nargs="*",
        help="network weights to merge into the model before training / 学習前にあらかじめモデルにマージするnetworkの重みファイル",
    )
    parser.add_argument(
        "--base_weights_multiplier",
        type=float,
        default=None,
        nargs="*",
        help="multiplier for network weights to merge into the model before training / 学習前にあらかじめモデルにマージするnetworkの重みの倍率",
    )
    parser.add_argument(
        "--no_half_vae",
        action="store_true",
        help="do not use fp16/bf16 VAE in mixed precision (use float VAE) / mixed precisionでも fp16/bf16 VAEを使わずfloat VAEを使う",
    )
    return parser

def check_cuda(observer:omo.OMObserver):
    gpu = GPUtil.getGPUs()[0]
    oml.debug(f"GPU=[{gpu.name}]")
    cuda_installed=torch.cuda.is_available()
    if cuda_installed: oml.success(f"Using CUDA with {gpu.name}")
    else: oml.warn(f"No GPU found! Using CPU instead.")
    oml.debug(f"PyTorch version: {torch.__version__}")
    observer.observe(observer.CUDA_INFO_EVENT,args=(gpu,cuda_installed))
    return
    
def print_args(args):
    argDict=vars(args)   
    oml.debug("************* START OF ARGS **********************")
    for argKey in argDict:
        oml.debug(f"{argKey}={argDict[argKey]}")
    oml.debug("************* END OF ARGS **********************")

def update_args(args,config_file,dataset_config_file,settings:omgs.OMGeneralSettings,hyper_parameters:omhp.OMHyperParameters):
    #s.root_dir,settings.project_name,s.model_file,s.model_cache_folder,
    platform_os = "WINDOWS" # TODO-> Do a smarter way
    args.in_json=config_file
    args.dataset_config=dataset_config_file
    # training config
    args.unet_lr = hyper_parameters.unet_lr #0.0005
    args.text_encoder_lr = hyper_parameters.text_encoder_lr #0.0001
    args.network_dim = hyper_parameters.network_dim #16
    args.network_alpha = hyper_parameters.network_alpha #8
    args.network_module = hyper_parameters.network_module #"networks.lora"
    # [optimizer_arguments]
    args.learning_rate = hyper_parameters.learning_rate #0.004 #0.001 (NOT BAD) # 0.000025 # MODIFIED 0.0005 # RECOMMEND BY HF = 0.000001
    args.optimizer_type = hyper_parameters.optimizer_name #"Adam" #"SGD" #"Adafactor" # MODIFIED AdamW8bit | Adafactor (TRY)
    args.lr_warmup_steps = hyper_parameters.lr_warmup_steps #65 # None for Adafactor otherwise around 65
    args.max_train_epochs = hyper_parameters.max_epochs # 8 # MODIFIED 10
    args.train_batch_size = hyper_parameters.batch_size # 2 # MODIFIED 2
    args.clip_skip = hyper_parameters.clip_skip # 2 # MODIFIED 2
    args.caption_extension=".txt"
    args.lr_scheduler = hyper_parameters.lr_scheduler #"cosine_with_restarts"
    args.lr_scheduler_num_cycles = hyper_parameters.lr_scheduler_num_cycles #3    
    # [training_arguments]
    args.save_every_n_epochs = settings.save_every_n_epochs #1
    args.save_last_n_epochs = settings.save_last_n_epochs #10
    args.min_snr_gamma = 5.0
    args.weighted_captions = settings.weighted_captions #False
    args.seed = settings.seed #42
    args.max_token_length = settings.max_token_length #225
    args.xformers = settings.xformers #True #True
    args.lowram = settings.lowram #False # False
    args.max_data_loader_n_workers = settings.max_data_loader_n_workers #0 #8
    args.persistent_data_loader_workers = settings.persistent_data_loader_workers #False #True
    if platform_os == "WINDOWS":
        args.max_data_loader_n_workers = 0 #8
        args.persistent_data_loader_workers = False #True
    args.save_precision = "fp16"
    args.mixed_precision = "fp16"
    args.output_dir = f"{settings.project_dir}/{settings.project_name}/output"
    args.logging_dir = f"{settings.project_dir}/_logs"
    args.output_name = settings.project_name
    args.log_prefix = settings.project_name
    # [model_arguments]
    args.pretrained_model_name_or_path = f"{settings.model_cache_folder}/{settings.model_file}"
    oml.debug(f"training with pretrained_model_name_or_path={args.pretrained_model_name_or_path}")
    args.v2 = settings.v2 #False
    # [saving_arguments]
    args.save_model_as = settings.save_model_as #"safetensors"
    # [dreambooth_arguments]
    args.prior_loss_weight = 1.0
    # [dataset_arguments]
    args.cache_latents = settings.cache_latents #True    
    # captions
    #args.recursive=False
    #args.train_data_dir=f"{root_dir}/{project_name}"

def train_network_main(observer:omo.OMObserver,settings,hyper_parameters,dataset_config_file, config_file, accelerate_config_file, num_cpu_threads_per_process):    
    check_cuda(observer)
    parser = setup_parser()
    args = parser.parse_args()         
    args = om_ext_train_util.read_config_from_file(args, parser)
    #set_default_args(args=args,config_file=config_file,dataset_config_file=dataset_config_file,root_dir=root_dir,project_name=project_name,model_file=model_file,model_folder=model_folder)
    update_args(args=args,config_file=config_file,dataset_config_file=dataset_config_file,settings=settings,hyper_parameters=hyper_parameters)

    #print_args(args)
    #make_captions(args) # make captions should be run from within the finetune folder as it is right now
    trainer = NetworkTrainer(observer)    
    trainer.train(args)
# endregion