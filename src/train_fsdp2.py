"""
Training Codes of LightningDiT together with VA-VAE using Hydra for configuration management.
It envolves advanced training methods, sampling methods, 
architecture design methods, computation methods. We achieve
state-of-the-art FID 1.35 on ImageNet 256x256.

by Maple (Jingfeng Yao) from HUST-VL
"""

import torch
# Enable TF32 for better performance on modern GPUs
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn

import torch.distributed as dist
import torch.nn as nn
from torch.distributed.fsdp import MixedPrecisionPolicy
from torch.distributed.distributed_c10d import ProcessGroup
from torch.distributed.checkpoint.state_dict import get_state_dict, set_state_dict, StateDictOptions
from torch.distributed.checkpoint.stateful import Stateful
import shutil
from torch.utils.data import DataLoader
import multiprocessing as mp
import warnings
import random
import json
import numpy as np
import logging
import os, io
from time import time
from copy import deepcopy
from collections import OrderedDict
from typing import List, Dict, Any, Optional
import datetime
import importlib
import hydra
import time
from omegaconf import DictConfig, OmegaConf
from collections import defaultdict
from torch.distributed.tensor import distribute_tensor, Replicate
import hydra
import time
from omegaconf import DictConfig, OmegaConf
from collections import defaultdict
from utils.train_utils import check_checkpoint_compatibility
from transport import Sampler
from utils.mm_logging import MMLogger
from utils.collect_env import collect_env
import wandb as wb
from tqdm import tqdm


_save_future = None  # Global to track pending save

def get_module_object(path):
    module_path, attribute = path.rsplit('.', 1)
    module = importlib.import_module(module_path)
    return getattr(module, attribute)


def setup_multi_processes(cfg):
    # set multi-process start method as `fork` to speed up the training
    mp_start_method = cfg.train.mp_start_method
    try:
        mp.set_start_method(mp_start_method)
    except RuntimeError:
        pass

    # setup OMP threads
    # This code is referred from https://github.com/pytorch/pytorch/blob/master/torch/distributed/run.py  # noqa
    if ('OMP_NUM_THREADS' not in os.environ and cfg.data.workers_per_gpu > 1):
        omp_num_threads = 1
        warnings.warn(
            f'Setting OMP_NUM_THREADS environment variable for each process '
            f'to be {omp_num_threads} in default, to avoid your system being '
            f'overloaded, please further tune the variable for optimal '
            f'performance in your application as needed.')
        os.environ['OMP_NUM_THREADS'] = str(omp_num_threads)

    # setup MKL threads
    if 'MKL_NUM_THREADS' not in os.environ and cfg.data.workers_per_gpu > 1:
        mkl_num_threads = 1
        warnings.warn(
            f'Setting MKL_NUM_THREADS environment variable for each process '
            f'to be {mkl_num_threads} in default, to avoid your system being '
            f'overloaded, please further tune the variable for optimal '
            f'performance in your application as needed.')
        os.environ['MKL_NUM_THREADS'] = str(mkl_num_threads)

def init_dist(cfg):
    world_size = int(os.environ['WORLD_SIZE'])
    rank = int(os.environ['RANK'])
    # LOCAL_RANK is set by `torch.distributed.launch` since PyTorch 1.1
    local_rank = int(os.environ['LOCAL_RANK'])
    num_devices_per_node = torch.cuda.device_count()
    timeout = cfg.dist_params.timeout
    if timeout is not None:
        timeout = datetime.timedelta(seconds=timeout)
    torch.cuda.set_device(local_rank)
    torch.distributed.init_process_group(
        backend="cpu:gloo,cuda:nccl",
        timeout=timeout
    )
    return world_size, rank, local_rank, num_devices_per_node

def sync_random_seed(group: Optional[ProcessGroup] = None) -> int:
    """Synchronize a random seed to all processes.

    In distributed sampling, different ranks should sample non-overlapped
    data in the dataset. Therefore, this function is used to make sure that
    each rank shuffles the data indices in the same order based
    on the same seed. Then different ranks could use different indices
    to select non-overlapped data from the same data list.

    Args:
        group (ProcessGroup, optional): The process group to work on. If None,
            the default process group will be used. Defaults to None.

    Returns:
        int: Random seed.

    Examples:
        >>> import torch
        >>> import mmengine.dist as dist

        >>> # non-distributed environment
        >>> seed = dist.sync_random_seed()
        >>> seed  # which a random number
        587791752

        >>> distributed environment
        >>> # We have 2 process groups, 2 ranks.
        >>> seed = dist.sync_random_seed()
        >>> seed
        587791752  # Rank 0
        587791752  # Rank 1
    """
    seed = np.random.randint(2**31)
    if dist.get_world_size() == 1:
        return seed

    if group is None:
        group = dist.distributed_c10d._get_default_group()

    backend_device = torch.device('cuda', torch.cuda.current_device())

    if dist.get_rank(group) == 0:
        random_num = torch.tensor(seed, dtype=torch.int32).to(backend_device)
    else:
        random_num = torch.tensor(0, dtype=torch.int32).to(backend_device)

    dist.broadcast(random_num, src=0, group=group)

    return random_num.item()

def set_random_seed(seed: Optional[int] = None,
                    deterministic: bool = False,
                    diff_rank_seed: bool = False) -> int:
    """Set random seed.

    Args:
        seed (int, optional): Seed to be used.
        deterministic (bool): Whether to set the deterministic option for
            CUDNN backend, i.e., set `torch.backends.cudnn.deterministic`
            to True and `torch.backends.cudnn.benchmark` to False.
            Defaults to False.
        diff_rank_seed (bool): Whether to add rank number to the random seed to
            have different random seed in different threads. Defaults to False.
    """
    if seed is None:
        seed = sync_random_seed()

    if diff_rank_seed:
        rank = dist.get_rank()
        seed += rank

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    
    # torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # os.environ['PYTHONHASHSEED'] = str(seed)
    if deterministic:
        if torch.backends.cudnn.benchmark:
            MMLogger.print_log(
                'torch.backends.cudnn.benchmark is going to be set as '
                '`False` to cause cuDNN to deterministically select an '
                'algorithm',
                logger='current',
                level=logging.WARNING)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        torch.use_deterministic_algorithms(True)
    return seed

def setup_wandb(cfg):
    os.environ['WANDB_DIR'] = cfg.train.wandb_dir
    os.environ['WANDB_CACHE_DIR'] = cfg.train.wandb_cache_dir
    config = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    wb.init(
        name=cfg.train.exp_name, 
        project=cfg.train.wandb_project, 
        config=config, 
        resume='allow',
    )
    
    
class AppState(Stateful):
    def __init__(self, model, optimizer=None, ema_dict=None, scheduler=None, logger=None):
        self.model = model
        self.optimizer = optimizer
        self.ema_dict = ema_dict
        self.scheduler = scheduler
        self.logger = logger
    def state_dict(self):
        model_state_dict, optimizer_state_dict = get_state_dict(self.model, self.optimizer)
        # Filter to only trainable params
        trainable_names = {n for n, p in self.model.named_parameters() if p.requires_grad}
        filtered_model_state = {
            k: v for k, v in model_state_dict.items() 
            if any(k.startswith(name) or name in k for name in trainable_names)
        }
        self.logger.info(f"Filtered model state: {filtered_model_state.keys()}")
        return {
            "model": filtered_model_state,
            "optim": optimizer_state_dict,
            "ema": {k: v.data for k, v in self.ema_dict.items()} if self.ema_dict else {},
            "scheduler": self.scheduler.state_dict() if self.scheduler else {},
        }

    def load_state_dict(self, state_dict):
        missing_keys, unexpected_keys = set_state_dict(
            self.model,
            self.optimizer,
            model_state_dict=state_dict["model"],
            optim_state_dict=state_dict["optim"],
            options=StateDictOptions(strict=False)
        )
        if len(missing_keys) > 0:
            self.logger.warning(f"Loading checkpoint: Missing keys: {missing_keys}")
        # Load EMA
        if self.ema_dict and "ema" in state_dict:
            for k, v in state_dict["ema"].items():
                if k in self.ema_dict:
                    self.ema_dict[k].copy_(v)
        # Load scheduler
        if self.scheduler and "scheduler" in state_dict:
            self.scheduler.load_state_dict(state_dict["scheduler"])
            
@hydra.main(version_base=None, config_path="configs", config_name="lightningdit_xl_vavae_f16d32")
def do_train(cfg: DictConfig):
    """
    Trains a LightningDiT using Hydra configuration management.
    """
    setup_multi_processes(cfg)
    if cfg.train.cudnn_benchmark:
        torch.backends.cudnn.benchmark = True
    world_size, rank, local_rank, num_devices_per_node = init_dist(cfg)
    master_process = rank == 0
    
    
    # Setup an experiment folder:
    experiment_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    checkpoint_dir = os.path.join(experiment_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = os.path.join(experiment_dir, f"{timestamp}.log")
    logger = MMLogger.get_instance('logger', log_file=log_file, distributed=True, log_level=cfg.train.log_level)
    logger.info(f"Experiment directory created at {experiment_dir}")
    if master_process:
        setup_wandb(cfg)
        
    meta = dict()
    # log env info
    try:
        env_info_dict = collect_env()
        env_info = '\n'.join([(f'{k}: {v}') for k, v in env_info_dict.items()])
        dash_line = '-' * 60 + '\n'
        logger.info('Environment info:\n' + dash_line + env_info + '\n' +
                    dash_line)
        meta['env_info'] = env_info
    except Exception as e:
        logger.warning(f'Error in collecting environment info: {e}')    
    config = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    config_str = json.dumps(config, indent=4)
    logger.info(f'Config:\n{config_str}')
    meta['config'] = config_str
    
    if cfg.train.global_seed is not None:
        logger.info(f'Set random seed to {cfg.train.global_seed}, '
                    f'deterministic: {cfg.train.deterministic}, '
                    f'use_rank_shift: {cfg.train.diff_seed}')
        cfg.train.global_seed = set_random_seed(
            cfg.train.global_seed,
            deterministic=cfg.train.deterministic,
            diff_rank_seed=cfg.train.diff_seed)
        
    meta['seed'] = cfg.train.global_seed
    meta['exp_name'] = os.path.basename(cfg.train.exp_name)

    
    # Create model:
    model = hydra.utils.instantiate(cfg.model)
    
    if cfg.train.weight_init is not None:
        init_path = cfg.train.weight_init
        logger.info(f"Loading pretrained weights from {init_path}")
        if os.path.exists(init_path) and os.path.isdir(init_path) and os.path.exists(os.path.join(init_path, ".metadata")):
            # DCP checkpoint (directory)
            missing_keys, unexpected_keys, common_keys = check_checkpoint_compatibility(init_path, model, app_name="app.ema")
            # Build nested structure matching AppState.state_dict() format
            model_sd = model.state_dict()
            state_dict = {
                "app": {
                    "ema": {k.replace("app.ema.", ""): model_sd[k.replace("app.ema.", "")].clone() for k in common_keys},
                }
            }
            
            # Load - DCP will automatically unshard
            torch.distributed.checkpoint.load(
                state_dict, 
                checkpoint_id=init_path,
                no_dist=True,  # Key: allows loading without distributed setup
            )
            missing_keys, unexpected_keys = model.load_state_dict(state_dict["app"]["ema"], strict=False)
        elif os.path.exists(init_path) and os.path.isfile(init_path) and init_path.endswith(".pt"):
            missing_keys, unexpected_keys = model.load_state_dict(torch.load(init_path), strict=False)
            
        if len(missing_keys) > 0:
            logger.warning(f"Missing keys: {missing_keys}")
        if len(unexpected_keys) > 0:
            logger.warning(f"Unexpected keys: {unexpected_keys}")
    
    
            
    device_mesh = dist.init_device_mesh("cuda", (world_size,))
    mp_policy = MixedPrecisionPolicy(
        param_dtype=getattr(torch, cfg.fsdp.param_dtype),
        reduce_dtype=getattr(torch, cfg.fsdp.reduce_dtype)
    )
    model = model.apply_fsdp2(
        device_mesh, 
        mp_policy,
        reshard_after_forward=cfg.fsdp.reshard_after_forward
    )
    ema_dict = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            ema_dict[name] = param.data.clone().detach()
    
    if cfg.optimizer.param_groups is not None:
        tags = cfg.optimizer.param_groups.tags
        lrs = cfg.optimizer.param_groups.lrs
        parameter_groups = {}
        null_param_group = {
            "params": [],
            "lr": cfg.optimizer.lr,
        }
        for tag, lr in zip(tags, lrs):
            param_group = {
                "params": [],
                "lr": lr,
            }
            if tag is not None:
                parameter_groups[tag] = param_group
        for name, param in model.named_parameters():
            if param.requires_grad:
                for tag in tags:
                    if tag in name:
                        parameter_groups[tag]["params"].append(param)
                        logger.info(f"LR: {parameter_groups[tag]['lr']}, Parameter: {name} with {param.numel() / 1e6:.2f}M parameters")
                        break
                else:   
                    null_param_group["params"].append(param)
                    logger.info(f"LR: {null_param_group['lr']}, Parameter: {name} with {param.numel() / 1e6:.2f}M parameters")
        parameter_groups = [null_param_group] + list(parameter_groups.values())
                    
    else:
        parameter_groups = [{"params": [], "lr": cfg.optimizer.lr}]
        for name, param in model.named_parameters():
            if param.requires_grad:
                parameter_groups[0]["params"].append(param)
                logger.info(f"LR: {parameter_groups[0]['lr']}, Parameter: {name} with {param.numel() / 1e6:.2f}M parameters")
            
    opt = torch.optim.AdamW(parameter_groups, weight_decay=0, betas=(0.9, cfg.optimizer.beta2))
    scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lambda step: min(1.0, cfg.optimizer.warmup_ratio + (1 - cfg.optimizer.warmup_ratio) * (step+1) / cfg.optimizer.warmup_steps))
    logger.info(f"Optimizer Parameters: {sum(p.numel() for param_group in parameter_groups for p in param_group['params']) / 1e6:.2f}M")
    train_steps = 0
    if cfg.train.resume is not None:
        resume_path = cfg.train.resume
        if os.path.exists(resume_path):
            # If pointing to checkpoint dir, find latest
            if os.path.isdir(resume_path) and not os.path.exists(os.path.join(resume_path, ".metadata")):
                # It's the parent checkpoints dir, find latest subdirectory
                subdirs = [d for d in os.listdir(resume_path) if os.path.isdir(os.path.join(resume_path, d))]
                if subdirs:
                    # find the latest checkpoint that contains .metadata
                    for subdir in sorted(subdirs)[::-1]:
                        if os.path.exists(os.path.join(resume_path, subdir, ".metadata")):
                            resume_path = os.path.join(resume_path, subdir)
                            break    
            
            train_steps = load_checkpoint(resume_path, model, opt, ema_dict, scheduler, logger)
        else:
            logger.info("No checkpoint found. Starting from scratch.")
    
    
    
    transport = hydra.utils.instantiate(cfg.transport)

    
    
    logger.info(f"LightningDiT Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    logger.info(f"Optimizer: AdamW, lr={cfg.optimizer.lr}, beta2={cfg.optimizer.beta2}")
    logger.info(f'Use lognorm sampling: {cfg.transport.use_lognorm}')
        
    # Setup data
    dataset = hydra.utils.instantiate(cfg.dataset, split_file=cfg.data.train_split)
    if cfg.data.sampler is not None:
        training_sampler = hydra.utils.instantiate(cfg.data.sampler, dataset=dataset, seed=cfg.train.global_seed, num_replicas=world_size, rank=local_rank)
        loader = DataLoader(
            dataset,
            num_workers=cfg.data.workers_per_gpu,
            collate_fn=dataset.collate_fn,
            batch_sampler=training_sampler,
            pin_memory=True,
        )
        logger.info(f"Dataset contains {len(dataset):,} images")
        logger.info(f"Using custom sampler {cfg.data.sampler}")
    else:
        training_sampler = torch.utils.data.distributed.DistributedSampler(dataset, num_replicas=world_size, rank=local_rank, shuffle=True)
    
        batch_size_per_gpu = int(np.round(cfg.train.global_batch_size / ( world_size * cfg.train.gradient_accumulation_steps)))
        global_batch_size = batch_size_per_gpu * world_size * cfg.train.gradient_accumulation_steps
        loader = DataLoader(
            dataset,
            batch_size=batch_size_per_gpu,
            num_workers=cfg.data.workers_per_gpu,
            collate_fn=dataset.collate_fn,
            sampler=training_sampler,
            pin_memory=True,
            drop_last=True,
        )
    
        logger.info(f"Dataset contains {len(dataset):,} images")
        logger.info(f"Batch size {batch_size_per_gpu} per gpu, with {global_batch_size} global batch size")

    valid_dataset = hydra.utils.instantiate(cfg.dataset, split_file=cfg.data.val_split, pad_everything=True)
    valid_sampler = torch.utils.data.distributed.DistributedSampler(valid_dataset, num_replicas=world_size, rank=local_rank, shuffle=False)
    valid_loader = DataLoader(valid_dataset, batch_size=4, num_workers=0, sampler=valid_sampler, drop_last=False)

    # Prepare models for training:
    update_ema(ema_dict, model, decay=0)
    model.train()
    
    log_steps = 0
    running_loss_dict = defaultdict(float)
    running_batch_size = 0
    start_time = time.time()
    train_iter = iter(loader)
    epoch = 0
    checkpoint_paths = []
    training_sampler.set_epoch(epoch)
    while train_steps < cfg.train.max_steps:
        
        for micro_step in range(cfg.train.gradient_accumulation_steps):
            is_last_microbatch = (micro_step == cfg.train.gradient_accumulation_steps - 1)
            try:
                data_dict = next(train_iter)
            except StopIteration:
                epoch += 1
                training_sampler.set_epoch(epoch)
                train_iter = iter(loader)
                data_dict = next(train_iter)
            
            x = data_dict["input"].to(local_rank)
            del data_dict["input"]
            data_dict = {k:v.to(local_rank) if isinstance(v, torch.Tensor) else v for k, v in data_dict.items()}
            model.set_requires_gradient_sync(is_last_microbatch)
            with torch.autocast(device_type="cuda", dtype=getattr(torch, cfg.fsdp.param_dtype)):
                loss_dict = transport.training_losses(model, x, data_dict)
            loss_dict = {k: v.mean() / cfg.train.gradient_accumulation_steps if "loss" in k else v / cfg.train.gradient_accumulation_steps for k, v in loss_dict.items() }
            for k, v in loss_dict.items():
                running_loss_dict[k] += v.item()
            loss = 0
            for k, v in loss_dict.items():
                if "loss" in k:
                    loss += v
            loss.backward()
            running_batch_size += data_dict.get("batch_size", x.shape[0])
        if cfg.optimizer.max_grad_norm > 0 and train_steps > cfg.optimizer.warmup_steps:
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.optimizer.max_grad_norm)
        else:
            grad_norm = torch.zeros(1).to(local_rank)

        opt.step()
        scheduler.step()
        opt.zero_grad(set_to_none=True)
        
        update_ema(ema_dict, model)
        train_steps += 1
        log_steps += 1

        
        if train_steps % cfg.train.log_every == 0:
            torch.cuda.synchronize()
            end_time = time.time()
            steps_per_sec = log_steps / (end_time - start_time)
            avg_loss_dict = {}
            total_batch_size = torch.tensor(running_batch_size).to(local_rank)
            dist.all_reduce(total_batch_size, op=dist.ReduceOp.SUM)
            total_batch_size = total_batch_size.item() / log_steps
            running_batch_size = 0
            for loss_name, loss_val in running_loss_dict.items():
                avg_loss = torch.tensor(loss_val / log_steps).to(local_rank)
                dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
                avg_loss = avg_loss.item() / dist.get_world_size()
                avg_loss_dict[loss_name] = avg_loss
                running_loss_dict[loss_name] = 0.0
            log_str = f"(step={train_steps:07d}) "
            for loss_name, loss_val in avg_loss_dict.items():
                log_str += f"{loss_name}: {loss_val:.4f}, "
            log_str += f"grad_norm: {grad_norm.item():.2f}, "
            log_str += f"batch_size: {total_batch_size:.2f}, "
            log_str += f"lr: {opt.param_groups[0]['lr']:.6f}, "
            log_str +=  f"Train Steps/Sec: {steps_per_sec:.2f}"
            logger.info(log_str)
            
            if master_process:
                log_dict = {
                    "train/steps_per_sec": steps_per_sec, 
                    "train/steps": train_steps, 
                    "train/lr": opt.param_groups[0]['lr'],
                    "train/grad_norm": grad_norm,
                }
                log_dict.update({f"train/{loss_name}": loss_val for loss_name, loss_val in avg_loss_dict.items()})
                wb.log(log_dict, step=train_steps)
            log_steps = 0
            start_time = time.time()

        # Save checkpoint:
        if train_steps % cfg.train.ckpt_every == 0:
            checkpoint_path = save_checkpoint(model, ema_dict, opt, scheduler, checkpoint_dir, train_steps, cfg, master_process, logger)
            if master_process:
                checkpoint_paths.append(checkpoint_path)
                if len(checkpoint_paths) > cfg.train.max_checkpoints:
                    ckpt_to_remove = checkpoint_paths.pop(0)
                    shutil.rmtree(ckpt_to_remove)
                    logger.info(f"Removed checkpoint {ckpt_to_remove}")
                    
        if train_steps % cfg.train.eval_every == 0:
            swap_ema(ema_dict, model)
            loss_dict = evaluate(model, transport, valid_loader, "cuda", getattr(torch, cfg.fsdp.param_dtype), local_rank)
            swap_ema(ema_dict, model)
            if master_process:
                log_dict = {
                    "eval/steps": train_steps,
                }
                log_dict.update({f"eval/{loss_name}": loss_val for loss_name, loss_val in loss_dict.items()})
                wb.log(log_dict, step=train_steps)
                json.dump(loss_dict, open(os.path.join(experiment_dir, f"loss_dict_{train_steps:07d}.json"), "w"))
            logger.info(f"Eval loss dict: {loss_dict}")

        if train_steps % cfg.train.sample_every == 0:
            swap_ema(ema_dict, model)
            sample_during_training(
                model, transport, valid_dataset, valid_loader, cfg,
                "cuda", getattr(torch, cfg.fsdp.param_dtype),
                logger, train_steps, experiment_dir, master_process,
            )
            swap_ema(ema_dict, model)

    if master_process:
        logger.info("Done!")
        
    torch.distributed.destroy_process_group()

    return True

@torch.no_grad()
def evaluate(model, transport, valid_loader, device, dtype, local_rank):
    running_loss_dict = defaultdict(float)
    n_samples = 0
    for data_dict in tqdm(valid_loader, total=len(valid_loader), desc="Evaluating"):
        data_dict = {k:v.to(device) if isinstance(v, torch.Tensor) else v for k, v in data_dict.items()}
        with torch.amp.autocast(device_type=device, dtype=dtype):
            output_dict = transport.training_losses(model, data_dict['input'], data_dict)
        output_dict = {k:v.mean().item() if "loss" in k else v.item() for k, v in output_dict.items()}
        for k, v in output_dict.items():
            running_loss_dict[k] += v
        n_samples += 1
    running_loss_dict = {k: v / n_samples for k, v in running_loss_dict.items()}
    avg_loss_dict = {}
    for loss_name, loss_val in running_loss_dict.items():
        avg_loss = torch.tensor(loss_val).to(local_rank)
        dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
        avg_loss = avg_loss.item() / dist.get_world_size()    
        avg_loss_dict[loss_name] = avg_loss
    return avg_loss_dict

def save_checkpoint(model, ema_dict, opt, scheduler, checkpoint_dir, train_steps, cfg, master_process, logger):
    checkpoint_path = f"{checkpoint_dir}/{train_steps:07d}"
    os.makedirs(checkpoint_path, exist_ok=True)
    
    global _save_future
    
    # Wait for previous save to complete
    if _save_future is not None:
        print(f"[Rank {dist.get_rank()}] Waiting for previous save to complete")
        _save_future.result()
    print(f"[Rank {dist.get_rank()}] Saving checkpoint to {checkpoint_path}")
    checkpoint_path = f"{checkpoint_dir}/{train_steps:07d}"
    os.makedirs(checkpoint_path, exist_ok=True)
    
    torch.cuda.empty_cache()
    
    app_state = AppState(model, opt, ema_dict, scheduler, logger)
    _save_future = torch.distributed.checkpoint.async_save({"app": app_state}, checkpoint_id=checkpoint_path)
    
    # Don't barrier here - let it be truly async
    return checkpoint_path

def load_checkpoint(checkpoint_path, model, opt=None, ema_dict=None, scheduler=None, logger=None):
    """Load checkpoint using DCP."""
    
    app_state = AppState(model, opt, ema_dict, scheduler, logger)
    torch.distributed.checkpoint.load({"app": app_state}, checkpoint_id=checkpoint_path)
    
    # Extract step number from path
    train_steps = int(os.path.basename(checkpoint_path))
    scheduler.last_epoch = train_steps
    logger.info(f"Resumed from checkpoint: {checkpoint_path}, step={train_steps}")
    return train_steps

@torch.no_grad()
def update_ema(ema_params, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """
    model_params = OrderedDict(model.named_parameters())
    for name, param in model_params.items():
        if param.requires_grad:
            ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)

@torch.no_grad()
def swap_ema(ema_dict, model):
    """Swap model parameters with EMA parameters. Call again to swap back.
    Works with FSDP2 DTensors by operating on local shards."""
    model_params = OrderedDict(model.named_parameters())
    for name, param in model_params.items():
        if name in ema_dict:
            # Get local tensor views to avoid DTensor/Tensor mixing
            param_local = param.data.to_local() if hasattr(param.data, 'to_local') else param.data
            ema_local = ema_dict[name].to_local() if hasattr(ema_dict[name], 'to_local') else ema_dict[name]
            tmp = param_local.clone()
            param_local.copy_(ema_local)
            ema_local.copy_(tmp)

@torch.no_grad()
def sample_during_training(model, transport, dataset, valid_loader, cfg, device, dtype, logger, train_steps, experiment_dir, master_process=False):
    """Generate samples during training and log to wandb.
    All ranks must participate (FSDP2 forward needs all ranks), but only master saves/logs."""
    model.eval()
    sampler = Sampler(transport)
    sample_fn = sampler.sample_ode(
        sampling_method=cfg.sample.sampling_method,
        num_steps=cfg.sample.num_sampling_steps,
        atol=cfg.sample.atol,
        rtol=cfg.sample.rtol,
        reverse=cfg.sample.reverse,
        timestep_shift=cfg.sample.timestep_shift,
    )

    # Get a batch of validation data (all ranks get same data via valid_loader)
    data_dict = next(iter(valid_loader))
    data_dict = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in data_dict.items()}
    z = torch.randn_like(data_dict["input"])

    with torch.amp.autocast(device_type="cuda", dtype=dtype):
        samples = sample_fn(z, model, **data_dict)

    # Only master reconstructs and logs
    if master_process:
        samples_np = samples[-1].cpu().numpy()
        data_dict_np = {k: v.cpu().numpy() if isinstance(v, torch.Tensor) else v for k, v in data_dict.items()}

        try:
            imgs, pts, panel_data, _, patterns, gt_patterns = dataset.reconstruct(samples_np, **data_dict_np)
            save_dir = os.path.join(experiment_dir, f"samples_{train_steps:07d}")
            os.makedirs(save_dir, exist_ok=True)
            wandb_images = []
            for k, img in enumerate(imgs):
                img.save(os.path.join(save_dir, f"sample_{k:03d}.png"))
                wandb_images.append(wb.Image(img, caption=f"sample_{k}"))
            wb.log({"samples": wandb_images}, step=train_steps)
            logger.info(f"Saved {len(imgs)} samples to {save_dir}")
        except Exception as e:
            logger.warning(f"Sample reconstruction failed: {e}")

    model.train()


if __name__ == "__main__":
    do_train()