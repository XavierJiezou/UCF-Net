import os
import logging

import torch.nn as nn


class RankFilter(logging.Filter):
    """Only allow log records from the specified DDP rank."""
    def __init__(self, rank=0):
        super().__init__()
        self.rank = rank

    def filter(self, record):
        rank = int(os.environ.get('LOCAL_RANK', 0))
        return rank == self.rank


def create_logger(log_path):
    log_dir = os.path.dirname(log_path)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.propagate = False

    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        try:
            handler.close()
        except Exception:
            pass

    formatter = logging.Formatter('[%(asctime)s] [%(levelname)-8s] %(message)s')

    fh = logging.FileHandler(log_path)
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setLevel(logging.INFO)
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    return logger


def _normalize_for_logging(obj):
    if isinstance(obj, dict):
        return {str(k): _normalize_for_logging(v) for k, v in sorted(obj.items(), key=lambda item: str(item[0]))}
    if isinstance(obj, (list, tuple)):
        return [_normalize_for_logging(v) for v in obj]
    if isinstance(obj, set):
        return [_normalize_for_logging(v) for v in sorted(obj, key=lambda x: str(x))]
    return obj


def config_to_pretty_dict(config):
    if isinstance(config, dict):
        return _normalize_for_logging(config)
    if hasattr(config, '__dict__'):
        return _normalize_for_logging(vars(config))
    return config


def count_parameters(module):
    total = sum(p.numel() for p in module.parameters())
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    return trainable, total


def count_named_parameters(model, predicate):
    params = [(name, param) for name, param in model.named_parameters() if predicate(name, param)]
    total = sum(param.numel() for _name, param in params)
    trainable = sum(param.numel() for _name, param in params if param.requires_grad)
    return trainable, total


def format_param_count(num_params):
    return f"{num_params / 1_000_000:.1f}M"


def format_param_count_with_exact(num_params):
    return f"{format_param_count(num_params)} ({num_params:,})"


def format_param_pair(trainable, total):
    return f"{format_param_count_with_exact(trainable)}/{format_param_count_with_exact(total)}"


def get_lr(optimizer):
    if optimizer is None or not optimizer.param_groups:
        return 0.0
    return optimizer.param_groups[0].get('lr', 0.0)


def _module_name(module):
    if module is None:
        return "N/A"
    return module.__class__.__name__


def _unwrap_model(model):
    return getattr(model, 'module', model)


def _append_param_line(lines, label, module):
    if module is None:
        return
    trainable, total = count_parameters(module)
    lines.append(f"{label} params (training/total): {format_param_pair(trainable, total)}")


def get_model_param_report(model):
    model = _unwrap_model(model)
    lines = []
    trainable, total = count_parameters(model)
    lines.append(f"Whole network params (training/total): {format_param_pair(trainable, total)}")

    if hasattr(model, 'clip_vision'):
        trainable, total = count_parameters(model.clip_vision)
        lines.append(
            f"CLIP Backbone params (training/total): {format_param_pair(trainable, total)}"
        )
    if hasattr(model, 'dino_model'):
        trainable, total = count_parameters(model.dino_model)
        lines.append(
            f"DINO Backbone params (training/total): {format_param_pair(trainable, total)}"
        )

    if hasattr(model, 'clip_vision') or hasattr(model, 'dino_model'):
        trainable, total = count_named_parameters(
            model,
            lambda name, _param: not (name.startswith('clip_vision.') or name.startswith('dino_model.')),
        )
        lines.append(f"Non-backbone params (training/total): {format_param_pair(trainable, total)}")

    _append_param_line(lines, "CLIP intra-branch Layer-MoE", getattr(model, 'clip_layer_moe', None))
    _append_param_line(lines, "DINO intra-branch Layer-MoE", getattr(model, 'dino_layer_moe', None))
    if hasattr(model, 'clip_layer_moe') or hasattr(model, 'dino_layer_moe'):
        moe_modules = [
            getattr(model, 'clip_layer_moe', nn.Identity()),
            getattr(model, 'dino_layer_moe', nn.Identity()),
        ]
        trainable, total = count_parameters(nn.ModuleList(moe_modules))
        lines.append(f"Intra-branch Layer-MoE total params (training/total): {format_param_pair(trainable, total)}")

    _append_param_line(lines, "Inter-branch fusion", getattr(model, 'fusion', None))
    _append_param_line(lines, "Classifier head", getattr(model, 'head', None))

    if hasattr(model, 'lora_enabled'):
        trainable, total = count_named_parameters(model, lambda name, _param: 'lora_' in name)
        lines.append(f"LoRA adapter params (training/total): {format_param_pair(trainable, total)}")

    if hasattr(model, 'clip_block_mlps'):
        clip_decoder = nn.ModuleList([
            model.clip_block_mlps,
            getattr(model, 'clip_aggregator', nn.Identity()),
            getattr(model, 'clip_moe', nn.Identity()),
            getattr(model, 'clip_mu_head', nn.Identity()),
            getattr(model, 'clip_logvar_head', nn.Identity()),
            getattr(model, 'clip_head', nn.Identity()),
        ])
        trainable, total = count_parameters(clip_decoder)
        lines.append(
            f"CLIP Decoder params (training/total): {format_param_pair(trainable, total)}"
        )
        lines.append(f"CLIP decode_head type: {_module_name(getattr(model, 'clip_head', None))}")

    if hasattr(model, 'dino_block_mlps'):
        dino_decoder = nn.ModuleList([
            model.dino_block_mlps,
            getattr(model, 'dino_aggregator', nn.Identity()),
            getattr(model, 'dino_moe', nn.Identity()),
            getattr(model, 'dino_mu_head', nn.Identity()),
            getattr(model, 'dino_logvar_head', nn.Identity()),
            getattr(model, 'dino_head', nn.Identity()),
        ])
        trainable, total = count_parameters(dino_decoder)
        lines.append(
            f"DINO Decoder params (training/total): {format_param_pair(trainable, total)}"
        )
        lines.append(f"DINO decode_head type: {_module_name(getattr(model, 'dino_head', None))}")

    top_level = []
    for child_name, child_module in model.named_children():
        if child_name == 'loss_func':
            continue
        trainable, total = count_parameters(child_module)
        if total == 0:
            continue
        top_level.append(f"{child_name}={format_param_pair(trainable, total)}")
    if top_level:
        lines.append("Top-level module params (training/total): " + "; ".join(top_level))

    return lines


def _format_scalar(value, digits=3):
    if value is None:
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return f"{value:.{digits}f}"


def _remap_metric_name(name):
    if name.startswith('loss_cls_clip'):
        return 'train/clip/loss_cls'
    if name.startswith('loss_cls_dino'):
        return 'train/dino/loss_cls'
    if name.startswith('loss_cls_fused'):
        return 'train/fused/loss_cls'
    if name.startswith('loss_kl_clip'):
        return 'train/clip/loss_kl'
    if name.startswith('loss_kl_dino'):
        return 'train/dino/loss_kl'
    if name.startswith('loss_kl_fused'):
        return 'train/fused/loss_kl'
    if name.startswith('loss_moe_balance_clip'):
        return 'train/clip/loss_moe_balance'
    if name.startswith('loss_moe_balance_dino'):
        return 'train/dino/loss_moe_balance'
    if name.startswith('loss_router_z_clip'):
        return 'train/clip/loss_router_z'
    if name.startswith('loss_router_z_dino'):
        return 'train/dino/loss_router_z'
    if name == 'loss_kl_branch_avg':
        return 'train/branch/loss_kl_avg'
    if name == 'loss_moe_balance':
        return 'train/branch/loss_moe_balance_avg'
    if name == 'loss_router_z':
        return 'train/branch/loss_router_z_avg'
    if name == 'overall':
        return 'train/loss_all'
    if name in ('acc', 'auc', 'eer', 'ap'):
        return f"train/{name}"
    return f"train/{name}"


def format_flat_train_log(iteration, iter_time, lr, loss_values, metric_values):
    pieces = [f"Iters: {iteration}", f"train/iter_time: {_format_scalar(iter_time)}", f"train/lr: {lr:.6f}"]

    for source in (loss_values, metric_values):
        for key in sorted(source.keys()):
            value = source[key]
            value_str = _format_scalar(value)
            if value_str is None:
                continue
            pieces.append(f"{_remap_metric_name(key)}: {value_str}")
    return ", ".join(pieces)


def format_eval_metrics(dataset_name, metric_dict):
    pieces = [f"dataset={dataset_name}"]
    for key in sorted(metric_dict.keys()):
        if key in ('pred', 'label', 'dataset_dict'):
            continue
        value_str = _format_scalar(metric_dict[key], digits=6)
        if value_str is None:
            continue
        pieces.append(f"{key}={value_str}")
    return ", ".join(pieces)
