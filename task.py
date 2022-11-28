# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""Creates lex-GLUE MCloud runs from config YAMLs"""

import os
import sys
from typing import cast
from omegaconf import OmegaConf as om, DictConfig

from torch.utils.data import DataLoader
from transformers import AutoConfig

from composer import Trainer, algorithms
from composer.core.types import Dataset
from composer.optim import DecoupledAdamW
from composer.optim.scheduler import (ConstantWithWarmupScheduler,
                                      CosineAnnealingWithWarmupScheduler,
                                      LinearWithWarmupScheduler)
from composer.utils import dist, reproducibility
from composer.callbacks import LRMonitor, MemoryMonitor, SpeedMonitor
from composer.loggers import WandBLogger
import wandb

from m_lex_glue.data import create_lexglue_dataset
from m_lex_glue.labels import TASK_NAME_TO_NUM_LABELS
from m_lex_glue.models.hf_model import get_huggingface_model


def build_dataloader(dataset, device_batch_size, **kwargs):
    import transformers
    dataset = cast(Dataset, dataset)

    return DataLoader(
        dataset=dataset,
        batch_size=device_batch_size,
        sampler=dist.get_sampler(dataset, drop_last=False, shuffle=False),
        collate_fn=transformers.default_data_collator,
        **kwargs,
    )


def build_logger(name, kwargs):
    if name == 'wandb':
        return WandBLogger(**kwargs)
    else:
        raise ValueError(f'Not sure how to build logger: {name}')


def build_callback(name, kwargs):
    if name == 'lr_monitor':
        return LRMonitor()
    elif name == 'memory_monitor':
        return MemoryMonitor()
    elif name == 'speed_monitor':
        return SpeedMonitor(window_size=kwargs.get('window_size', 1))
    else:
        raise ValueError(f'Not sure how to build callback: {name}')


def build_algorithm(name, kwargs):
    if name == 'fused_layernorm':
        return algorithms.FusedLayerNorm(**kwargs)
    else:
        raise ValueError(f'Not sure how to build algorithm: {name}')


def build_optimizer(cfg, model):
    if cfg.name == 'decoupled_adamw':
        return DecoupledAdamW(
            model.parameters(),
            lr=cfg.lr,
            betas=cfg.betas,
            eps=cfg.eps,
            weight_decay=cfg.weight_decay
        )
    else:
        raise ValueError(f'Not sure how to build optimizer: {cfg.name}')

def build_scheduler(cfg):
    if cfg.name == 'constant_with_warmup':
        return ConstantWithWarmupScheduler(
            t_warmup=cfg.t_warmup)
    elif cfg.name == 'linear_decay_with_warmup':
        return LinearWithWarmupScheduler(
            t_warmup=cfg.t_warmup,
            alpha_f=cfg.alpha_f
        )
    elif cfg.name == 'cosine_with_warmup':
        return CosineAnnealingWithWarmupScheduler(
            t_warmup=cfg.t_warmup,
            alpha_f=cfg.alpha_f)
    else:
        raise ValueError(f'Not sure how to build scheduler: {cfg.name}')


def build_model(cfg: DictConfig, task_name: str):
    config = AutoConfig.from_pretrained(
        cfg.model_name,
        num_labels=TASK_NAME_TO_NUM_LABELS[task_name],
        finetuning_task=task_name,
    )
    return get_huggingface_model(cfg, config)


def main(task_name: str, cfg: DictConfig) -> None:
    print("Training using config: ")
    print(om.to_yaml(cfg, resolve=True))
    reproducibility.seed_all(cfg.seed)

    # Build Model
    print('Initializing model...')
    model = build_model(cfg, task_name)
    n_params = sum(p.numel() for p in model.parameters())
    print(f'{n_params=:.4e}')

    # Get batch size info
    device_train_batch_size = cfg.global_train_batch_size // dist.get_world_size()
    device_eval_batch_size = cfg.get('global_eval_batch_size', cfg.global_train_batch_size) // dist.get_world_size()

    # Dataloaders
    print("Building eval loader...")
    eval_dataset = create_lexglue_dataset(task_name, model.tokenizer, split="validation", max_seq_length=cfg.max_seq_length)
    eval_loader = build_dataloader(eval_dataset, device_eval_batch_size)
    print("Building train loader...")
    train_dataset = create_lexglue_dataset(task_name, model.tokenizer, split="train", max_seq_length=cfg.max_seq_length)
    train_loader = build_dataloader(train_dataset, device_train_batch_size)

    # Optimizer
    optimizer = build_optimizer(cfg.optimizer, model)
    
    # Scheduler
    scheduler = build_scheduler(cfg.scheduler)

    # Loggers
    loggers = [build_logger(name, logger_cfg) for name, logger_cfg in cfg.get('loggers', {}).items()]

    # Callbacks
    callbacks = [build_callback(name, callback_cfg) for name, callback_cfg in cfg.get('callbacks', {}).items()]

    # Algorithms
    algorithms = [build_algorithm(name, algorithm_cfg) for name, algorithm_cfg in cfg.get('algorithms', {}).items()]

    if 'run_name' in cfg:
        run_name = cfg['run_name']
    else:
        run_name = os.environ['COMPOSER_RUN_NAME']

    # Build the Trainer
    trainer = Trainer(
        run_name=run_name,
        seed=cfg.seed,
        model=model,
        algorithms=algorithms,
        train_dataloader=train_loader,
        eval_dataloader=eval_loader,
        train_subset_num_batches=cfg.get('train_subset_num_batches', -1),
        eval_subset_num_batches=cfg.get('eval_subset_num_batches', -1),
        optimizers=optimizer,
        schedulers=scheduler,
        max_duration=cfg.max_duration,
        eval_interval=cfg.eval_interval,
        progress_bar=cfg.progress_bar,
        log_to_console=cfg.log_to_console,
        loggers=loggers,
        callbacks=callbacks,
        precision=cfg.precision,
        device=cfg.get('device', None),
        grad_clip_norm=cfg.grad_clip_norm,
        grad_accum=cfg.get('grad_accum', 'auto'),
        save_folder=cfg.get('save_folder', None),
        save_interval=cfg.get('save_interval', '1000ba'),
        save_num_checkpoints_to_keep=cfg.get('save_num_checkpoints_to_keep', -1),
        load_path=cfg.get('load_path', None),
        load_weights_only=cfg.get('load_weights_only', False),
    )

    print("Logging config...")
    config_dict: Dict[str, Any] = om.to_container(cfg, resolve=True)  # type: ignore
    config_dict.update({
        'n_gpus': dist.get_world_size(),
        'n_params': n_params,
        'device_train_batch_size': device_train_batch_size,
        'device_eval_batch_size': device_eval_batch_size,
    })
    if wandb.run is not None:
        wandb.config.update(config_dict)

    print("Starting training...")
    trainer.fit()


if __name__ == '__main__':
    yaml_path, args_list = sys.argv[1], sys.argv[2:]
    with open(yaml_path) as f:
        yaml_cfg = om.load(f)
    cli_cfg = om.from_cli(args_list)
    cfg: DictConfig = om.merge(yaml_cfg, cli_cfg)  # type: ignore
    task = cfg.task
    main(task, cfg)