import os
import argparse
from os.path import join
import cv2
import random
import datetime
import time
import yaml
import json
import math
import re
from tqdm import tqdm
import numpy as np
from datetime import timedelta
from copy import deepcopy
from PIL import Image as pil_image

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.utils.data
import torch.optim as optim
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist

from optimizor.SAM import SAM
from optimizor.LinearLR import LinearDecayLR

from trainer.trainer import Trainer
from detectors import DETECTOR
from dataset import *
from logger import (
    create_logger,
    RankFilter,
    config_to_pretty_dict,
    get_model_param_report,
    get_lr,
    format_eval_metrics,
)


parser = argparse.ArgumentParser(description='Process some paths.')
parser.add_argument('--detector_path', type=str,
                    default='./training/config/detector/ucfnet.yaml',
                    help='path to detector YAML file')
parser.add_argument('--train_config_path', type=str,
                    default='./training/config/train_config.yaml',
                    help='path to base train YAML file')
parser.add_argument("--train_dataset", nargs="+")
parser.add_argument("--test_dataset", nargs="+")
parser.add_argument('--no-save_ckpt', dest='save_ckpt', action='store_false', default=None)
parser.add_argument('--no-save_feat', dest='save_feat', action='store_false', default=None)
parser.add_argument('--resume', type=str, default=None,
                    help='path to a latest.pth checkpoint to resume training')
parser.add_argument('--resume-weights-only', action='store_true', default=False,
                    help='only load model weights from --resume, without optimizer/scheduler/epoch state')
parser.add_argument("--ddp", action='store_true', default=False)
parser.add_argument('--local_rank', '--local-rank', dest='local_rank', type=int, default=0)
args = parser.parse_args()
torch.cuda.set_device(args.local_rank)


def count_json_items(json_path):
    if not json_path:
        return None
    with open(json_path, 'r') as f:
        data = json.load(f)
    if isinstance(data, dict) and isinstance(data.get('items'), list):
        return len(data['items'])
    if isinstance(data, list):
        return len(data)
    return None


def format_dataset_size_summary(config):
    image_json_config = config.get('image_data_json') or {}
    if image_json_config.get('enabled', False):
        train_path = image_json_config.get('train')
        val_path = image_json_config.get('val')
        train_count = count_json_items(train_path)
        val_count = count_json_items(val_path)
        return (
            f"Dataset sizes: train={train_count} ({train_path}), "
            f"val={val_count} ({val_path})"
        )
    return None


def format_best_metric_summary(trainer, metric_scoring):
    best_record = trainer.best_metrics_all_time.get('avg', {})
    default_value = float('inf') if metric_scoring == 'eer' else 0.0
    best_value = best_record.get(metric_scoring, default_value)
    best_epoch = best_record.get('epoch', 'N/A')
    return f"Best {metric_scoring.upper()}(epoch {best_epoch}): {best_value:.6f}"


def init_seed(config):
    if config['manualSeed'] is None:
        config['manualSeed'] = random.randint(1, 10000)
    random.seed(config['manualSeed'])
    if config['cuda']:
        torch.manual_seed(config['manualSeed'])
        torch.cuda.manual_seed_all(config['manualSeed'])


def build_dataloader_kwargs(config, batch_size, shuffle=None, sampler=None, drop_last=False):
    num_workers = int(config['workers'])
    loader_kwargs = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=None,
        drop_last=drop_last,
    )
    if shuffle is not None:
        loader_kwargs['shuffle'] = shuffle
    if sampler is not None:
        loader_kwargs['sampler'] = sampler

    if num_workers > 0:
        loader_kwargs['pin_memory'] = bool(config.get('pin_memory', True))
        loader_kwargs['persistent_workers'] = bool(config.get('persistent_workers', True))
        loader_kwargs['prefetch_factor'] = int(config.get('prefetch_factor', 4))

    return loader_kwargs


class DistributedEvalSampler(torch.utils.data.Sampler):
    """Shard evaluation exactly across ranks without padding duplicate samples."""

    def __init__(self, dataset):
        self.dataset = dataset
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        self.indices = list(range(self.rank, len(dataset), self.world_size))

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)


def prepare_training_data(config):
    # Only use the blending dataset class in training
    train_set = DeepfakeAbstractBaseDataset(
        config=config,
        mode='train',
    )
    if config['ddp']:
        sampler = DistributedSampler(train_set)
        loader_kwargs = build_dataloader_kwargs(
            config=config,
            batch_size=config['train_batchSize'],
            sampler=sampler,
            drop_last=bool(config.get('train_drop_last', False)),
        )
        loader_kwargs['collate_fn'] = train_set.collate_fn
        train_data_loader = torch.utils.data.DataLoader(
            dataset=train_set,
            **loader_kwargs,
        )
    else:
        loader_kwargs = build_dataloader_kwargs(
            config=config,
            batch_size=config['train_batchSize'],
            shuffle=True,
            drop_last=bool(config.get('train_drop_last', False)),
        )
        loader_kwargs['collate_fn'] = train_set.collate_fn
        train_data_loader = torch.utils.data.DataLoader(
            dataset=train_set,
            **loader_kwargs,
        )
    return train_data_loader


def prepare_testing_data(config):
    def get_test_data_loader(config, test_name):
        # update the config dictionary with the specific testing dataset
        config = config.copy()  # create a copy of config to avoid altering the original one
        config['test_dataset'] = test_name  # specify the current test dataset

        test_set = DeepfakeAbstractBaseDataset(
                config=config,
                mode='test',
        )

        sampler = DistributedEvalSampler(test_set) if config['ddp'] else None
        loader_kwargs = build_dataloader_kwargs(
            config=config,
            batch_size=config['test_batchSize'],
            shuffle=False if sampler is None else None,
            sampler=sampler,
            drop_last=False if sampler is not None else (test_name == 'DeepFakeDetection'),
        )
        loader_kwargs['collate_fn'] = test_set.collate_fn
        test_data_loader = torch.utils.data.DataLoader(
            dataset=test_set,
            **loader_kwargs,
        )

        return test_data_loader

    test_data_loaders = {}
    for one_test_name in config['test_dataset']:
        test_data_loaders[one_test_name] = get_test_data_loader(config, one_test_name)
    return test_data_loaders


def choose_optimizer(model, config):
    opt_name = config['optimizer']['type']
    if opt_name == 'sgd':
        optimizer = optim.SGD(
            params=model.parameters(),
            lr=config['optimizer'][opt_name]['lr'],
            momentum=config['optimizer'][opt_name]['momentum'],
            weight_decay=config['optimizer'][opt_name]['weight_decay']
        )
        return optimizer
    elif opt_name == 'adam':
        optimizer = optim.Adam(
            params=model.parameters(),
            lr=config['optimizer'][opt_name]['lr'],
            weight_decay=config['optimizer'][opt_name]['weight_decay'],
            betas=(config['optimizer'][opt_name]['beta1'], config['optimizer'][opt_name]['beta2']),
            eps=config['optimizer'][opt_name]['eps'],
            amsgrad=config['optimizer'][opt_name]['amsgrad'],
        )
        return optimizer
    elif opt_name == 'adamw':
        adamw_cfg = config['optimizer'][opt_name]
        optimizer = optim.AdamW(
            params=model.parameters(),
            lr=adamw_cfg['lr'],
            weight_decay=adamw_cfg['weight_decay'],
            betas=(adamw_cfg.get('beta1', 0.9), adamw_cfg.get('beta2', 0.999)),
            eps=adamw_cfg.get('eps', 1e-8),
            amsgrad=adamw_cfg.get('amsgrad', False),
        )
        return optimizer
    elif opt_name == 'sam':
        optimizer = SAM(
            model.parameters(),
            optim.SGD,
            lr=config['optimizer'][opt_name]['lr'],
            momentum=config['optimizer'][opt_name]['momentum'],
        )
    else:
        raise NotImplementedError('Optimizer {} is not implemented'.format(config['optimizer']))
    return optimizer


def choose_scheduler(config, optimizer, steps_per_epoch=None):
    if config['lr_scheduler'] is None:
        return None
    elif config['lr_scheduler'] == 'step':
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=config['lr_step'],
            gamma=config['lr_gamma'],
        )
        return scheduler
    elif config['lr_scheduler'] == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config['lr_T_max'],
            eta_min=config['lr_eta_min'],
        )
        return scheduler
    elif config['lr_scheduler'] == 'linear':
        scheduler = LinearDecayLR(
            optimizer,
            config['nEpochs'],
            int(config['nEpochs']/4),
        )
    elif config['lr_scheduler'] == 'warmup_cosine':
        steps_per_epoch = max(1, int(steps_per_epoch or 1))
        start_epoch = int(config.get('start_epoch', 0))
        total_epochs = max(1, int(config['nEpochs']) - start_epoch)
        if config.get('swa_enabled', False):
            swa_start_epoch = int(config.get('swa_start_epoch', max(1, int(config['nEpochs']) // 2)))
            total_epochs = max(1, min(total_epochs, swa_start_epoch - start_epoch))
        total_steps = max(1, total_epochs * steps_per_epoch)
        warmup_steps = max(1, int(config.get('warmup_epochs', 1)) * steps_per_epoch)
        base_lr = float(config['optimizer'][config['optimizer']['type']]['lr'])
        eta_min = float(config.get('lr_eta_min', 0.0))
        min_factor = eta_min / base_lr if base_lr > 0 else 0.0

        def lr_lambda(current_step):
            if current_step < warmup_steps:
                return max(min_factor, float(current_step + 1) / float(warmup_steps))
            denom = max(1, total_steps - warmup_steps)
            progress = min(1.0, float(current_step - warmup_steps) / float(denom))
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            return min_factor + (1.0 - min_factor) * cosine

        config['scheduler_step_unit'] = 'iteration'
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    else:
        raise NotImplementedError('Scheduler {} is not implemented'.format(config['lr_scheduler']))
    config.setdefault('scheduler_step_unit', 'epoch')
    return scheduler


def choose_metric(config):
    metric_scoring = config['metric_scoring']
    if metric_scoring not in ['eer', 'auc', 'acc', 'ap']:
        raise NotImplementedError('metric {} is not implemented'.format(metric_scoring))
    return metric_scoring


def unwrap_model(model):
    return model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model


def load_model_state(model, state_dict, strict=True):
    target_model = unwrap_model(model)
    try:
        return target_model.load_state_dict(state_dict, strict=strict)
    except RuntimeError:
        has_module_prefix = any(key.startswith('module.') for key in state_dict.keys())
        target_has_module_prefix = any(key.startswith('module.') for key in target_model.state_dict().keys())
        if has_module_prefix and not target_has_module_prefix:
            state_dict = {key[len('module.'):]: value for key, value in state_dict.items()}
        elif target_has_module_prefix and not has_module_prefix:
            state_dict = {f'module.{key}': value for key, value in state_dict.items()}
        return target_model.load_state_dict(state_dict, strict=strict)


def _is_better_metric(value, best_value, metric_scoring):
    if metric_scoring == 'eer':
        return value < best_value
    return value > best_value


def _parse_metric_log_line(line):
    if 'dataset=' not in line:
        return None, {}

    metrics = {}
    dataset_key = None
    for match in re.finditer(r'([A-Za-z_][\w/]*)=([^,\s]+)', line):
        key = match.group(1)
        value = match.group(2)
        if key == 'dataset':
            dataset_key = value
            continue
        try:
            metrics[key] = float(value)
        except ValueError:
            continue

    epoch_match = re.search(r'Epoch\[(\d+)\]', line)
    if epoch_match and 'epoch' not in metrics:
        metrics['epoch'] = int(epoch_match.group(1))

    return dataset_key, metrics


def load_best_metrics_from_log(log_path, metric_scoring):
    if not os.path.isfile(log_path):
        return {}

    default_best = float('inf') if metric_scoring == 'eer' else float('-inf')
    best_metrics = {}
    with open(log_path, 'r') as f:
        for line in f:
            dataset_key, metrics = _parse_metric_log_line(line)
            if not dataset_key or metric_scoring not in metrics:
                continue
            best_value = best_metrics.get(dataset_key, {}).get(metric_scoring, default_best)
            has_better_score = _is_better_metric(metrics[metric_scoring], best_value, metric_scoring)
            has_better_metadata = (
                metrics[metric_scoring] == best_value
                and 'epoch' in metrics
                and 'epoch' not in best_metrics.get(dataset_key, {})
            )
            if has_better_score or has_better_metadata:
                best_metrics[dataset_key] = dict(metrics)

    return best_metrics


def load_resume_checkpoint(config, trainer, checkpoint_path, resume_weights_only=False):
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Resume checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    state_dict = checkpoint.get('state_dict', checkpoint) if isinstance(checkpoint, dict) else checkpoint
    load_model_state(trainer.model, state_dict, strict=True)

    resumed_epoch = None
    resumed_iteration = None
    if isinstance(checkpoint, dict):
        resumed_epoch = checkpoint.get('epoch')
        resumed_iteration = checkpoint.get('iteration')

        if not resume_weights_only:
            if checkpoint.get('optimizer') is not None:
                trainer.optimizer.load_state_dict(checkpoint['optimizer'])
            if trainer.scheduler is not None and checkpoint.get('scheduler') is not None:
                trainer.scheduler.load_state_dict(checkpoint['scheduler'])

        best_metrics = checkpoint.get('best_metrics_all_time')
        if isinstance(best_metrics, dict):
            for dataset_key, metric_values in best_metrics.items():
                if isinstance(metric_values, dict):
                    trainer.best_metrics_all_time[dataset_key].update(metric_values)

    log_best_metrics = load_best_metrics_from_log(
        os.path.join(os.path.dirname(checkpoint_path), 'training.log'),
        trainer.metric_scoring,
    )
    recovered_keys = []
    default_best = float('inf') if trainer.metric_scoring == 'eer' else float('-inf')
    for dataset_key, metric_values in log_best_metrics.items():
        current_best = trainer.best_metrics_all_time[dataset_key].get(
            trainer.metric_scoring,
            default_best,
        )
        log_best = metric_values[trainer.metric_scoring]
        if _is_better_metric(log_best, current_best, trainer.metric_scoring):
            trainer.best_metrics_all_time[dataset_key].update(metric_values)
            recovered_keys.append(dataset_key)
    if recovered_keys:
        trainer.logger.info(
            f"Recovered best metrics from training.log for datasets: "
            f"{', '.join(sorted(recovered_keys))}"
        )

    if not resume_weights_only and resumed_epoch is not None:
        config['start_epoch'] = int(resumed_epoch) + 1

    trainer.logger.info(
        f"Resumed checkpoint: path={checkpoint_path}, "
        f"epoch={resumed_epoch}, iteration={resumed_iteration}, "
        f"next_start_epoch={config.get('start_epoch')}, "
        f"weights_only={resume_weights_only}"
    )


def build_run_dir(config, time_now):
    if 'task_target' not in config:
        run_name = f"{config['model_name']}_{time_now}"
    else:
        task_str = f"_{config['task_target']}" if config['task_target'] is not None else ""
        run_name = f"{config['model_name']}{task_str}_{time_now}"
    return os.path.join(config['log_dir'], run_name), run_name


def main():
    # parse options and load config
    with open(args.detector_path, 'r') as f:
        config = yaml.safe_load(f)
    with open(args.train_config_path, 'r') as f:
        config2 = yaml.safe_load(f)
    # Merge: train_config provides defaults, detector config overrides
    detector_overrides = dict(config)
    config.update(config2)
    config.update({k: v for k, v in detector_overrides.items() if v is not None})
    config['local_rank']=args.local_rank
    if config['dry_run']:
        config['nEpochs'] = 0
        config['save_feat']=False
    # If arguments are provided, they will overwrite the yaml settings
    if args.train_dataset:
        config['train_dataset'] = args.train_dataset
    if args.test_dataset:
        config['test_dataset'] = args.test_dataset
    if args.save_ckpt is not None:
        config['save_ckpt'] = args.save_ckpt
    if args.save_feat is not None:
        config['save_feat'] = args.save_feat
    if config['lmdb']:
        config['dataset_json_folder'] = 'preprocessing/dataset_json_v3'

    resume_path = os.path.abspath(args.resume) if args.resume else None
    if resume_path:
        config['resume_checkpoint'] = resume_path
        config['resume_weights_only'] = args.resume_weights_only
    if resume_path and config.get('lr_scheduler') == 'warmup_cosine':
        # Build the LambdaLR with the original full training horizon before loading
        # its saved state. start_epoch is advanced after checkpoint loading.
        config['start_epoch'] = 0

    time_now = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    if resume_path:
        run_dir = os.path.dirname(resume_path)
        run_name = os.path.basename(run_dir)
        prefix = f"{config['model_name']}_"
        if run_name.startswith(prefix):
            time_now = run_name[len(prefix):]
    else:
        run_dir, run_name = build_run_dir(config, time_now)
    os.makedirs(run_dir, exist_ok=True)
    logger = create_logger(os.path.join(run_dir, 'training.log'))
    config['ddp']= args.ddp
    if config['ddp']:
        logger.addFilter(RankFilter(0))
    if args.local_rank == 0:
        with open(os.path.join(run_dir, 'config.yaml'), 'w') as f:
            yaml.safe_dump(config_to_pretty_dict(config), f, sort_keys=False)
    logger.info(f"Save log to {run_dir}")
    if args.local_rank == 0:
        dataset_size_summary = format_dataset_size_summary(config)
        if dataset_size_summary is not None:
            logger.info(dataset_size_summary)
    logger.info(config_to_pretty_dict(config))

    # init seed
    init_seed(config)

    # set cudnn benchmark if needed
    if config['cudnn']:
        cudnn.benchmark = True
    if config['ddp']:
        ddp_timeout_minutes = int(config.get('ddp_timeout_minutes', 720))
        dist.init_process_group(
            backend='nccl',
            timeout=timedelta(minutes=ddp_timeout_minutes)
        )
    # prepare the training data loader
    train_data_loader = prepare_training_data(config)

    # prepare the testing data loader
    test_data_loaders = prepare_testing_data(config)

    # prepare the model (detector)
    model_class = DETECTOR[config['model_name']]
    model = model_class(config)
    for line in get_model_param_report(model):
        logger.info(line)

    # prepare the optimizer
    optimizer = choose_optimizer(model, config)

    # prepare the scheduler
    scheduler = choose_scheduler(config, optimizer, steps_per_epoch=len(train_data_loader))

    # prepare the metric
    metric_scoring = choose_metric(config)

    # prepare the trainer
    trainer = Trainer(config, model, optimizer, scheduler, logger, metric_scoring, time_now=time_now)
    trainer.log_dir = run_dir
    trainer.run_name = run_name

    if resume_path:
        load_resume_checkpoint(
            config=config,
            trainer=trainer,
            checkpoint_path=resume_path,
            resume_weights_only=args.resume_weights_only,
        )
        if args.local_rank == 0:
            with open(os.path.join(run_dir, 'config.yaml'), 'w') as f:
                yaml.safe_dump(config_to_pretty_dict(config), f, sort_keys=False)

    # ── SWA setup (real integration, not placeholder) ──
    swa_model = None
    swa_scheduler = None
    swa_enabled = config.get('swa_enabled', False)
    swa_start = config.get('swa_start_epoch', max(1, config['nEpochs'] // 2))
    if swa_enabled:
        from torch.optim.swa_utils import AveragedModel, SWALR
        model_for_swa = trainer.model.module if isinstance(trainer.model, torch.nn.parallel.DistributedDataParallel) else trainer.model
        swa_device = next(model_for_swa.parameters()).device
        swa_model = AveragedModel(model_for_swa, device=swa_device)
        trainer.swa_model = swa_model
        swa_lr = config.get('swa_lr', 0.0001)
        swa_scheduler = SWALR(optimizer, swa_lr=swa_lr)
        logger.info(f"SWA enabled: start_epoch={swa_start}, swa_lr={swa_lr}")

    total_epochs = max(0, config['nEpochs'] - config['start_epoch'])
    total_iterations = len(train_data_loader) * total_epochs
    logger.info(f"Run name: {os.path.basename(trainer.log_dir)}")
    logger.info(f"Train for {total_epochs} epochs / {total_iterations} iterations.")

    # start training
    for epoch in range(config['start_epoch'], config['nEpochs']):
        current_lr = get_lr(optimizer)
        logger.info(
            f"===========> Epoch: {epoch}, LR: {current_lr:.6f}, "
            f"{format_best_metric_summary(trainer, metric_scoring)}"
        )
        trainer.model.epoch = epoch
        best_metric = trainer.train_epoch(
                    epoch=epoch,
                    train_data_loader=train_data_loader,
                    test_data_loaders=test_data_loaders,
                )
        if best_metric is not None:
            if isinstance(best_metric, dict) and 'current' in best_metric and 'best' in best_metric:
                current_metric = best_metric.get('current') or {}
                best_metric_all = best_metric.get('best') or {}
                current_avg = current_metric.get('avg', {}) if isinstance(current_metric, dict) else {}
                best_avg = best_metric_all.get('avg', {}) if isinstance(best_metric_all, dict) else {}
                if current_avg:
                    logger.info(f"Epoch[{epoch}] current eval, {format_eval_metrics('avg', current_avg)}")
                if best_avg:
                    logger.info(f"Epoch[{epoch}] best eval, {format_eval_metrics('avg', best_avg)}")
                if not current_avg and not best_avg:
                    logger.info(f"Epoch[{epoch}] evaluation finished, but no eval metrics were returned in this epoch.")
            else:
                logger.info(f"Epoch[{epoch}] evaluation finished.")

        # ── SWA: update averaged model after swa_start_epoch ──
        if swa_enabled and epoch >= swa_start:
            model_for_swa = trainer.model.module if isinstance(trainer.model, torch.nn.parallel.DistributedDataParallel) else trainer.model
            swa_model.update_parameters(model_for_swa)
            swa_scheduler.step()
            logger.info(f"SWA updated at epoch {epoch}")
        elif scheduler is not None and config.get('scheduler_step_unit', 'epoch') == 'epoch':
            scheduler.step()

    # ── SWA: swap to averaged weights and update BN ──
    if swa_enabled and swa_model is not None:
        logger.info("SWA: swapping to averaged weights and updating BN...")
        torch.optim.swa_utils.update_bn(train_data_loader, swa_model, device=next(model.parameters()).device)
        # Replace trainer model with SWA model for final eval/save
        trainer.model = swa_model.module
        # Save SWA checkpoint
        if config.get('local_rank', 0) == 0:
            swa_ckpt_path = os.path.join(trainer.log_dir, 'swa_model.pth')
            swa_payload = {
                'state_dict': swa_model.module.state_dict(),
                'ckpt_info': f"swa_epoch_{config['nEpochs'] - 1}",
                'phase': 'swa',
                'dataset_key': 'swa',
            }
            torch.save(swa_payload, swa_ckpt_path)
            logger.info(f"SWA checkpoint saved to {swa_ckpt_path}")

    final_metric_to_print = best_metric['best'] if isinstance(best_metric, dict) and 'best' in best_metric else best_metric
    if isinstance(final_metric_to_print, dict) and 'avg' in final_metric_to_print:
        logger.info(f"Stop Training on best metric, {format_eval_metrics('avg', final_metric_to_print['avg'])}")
    else:
        logger.info("Stop Training.")
    # update
    if 'svdd' in config['model_name']:
        model.update_R(epoch)
    # close the tensorboard writers
    for writer in trainer.writers.values():
        writer.close()



if __name__ == '__main__':
    main()
