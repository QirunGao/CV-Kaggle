import random
import numpy as np
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import (
    CosineAnnealingLR, StepLR, MultiStepLR, ExponentialLR, ReduceLROnPlateau
)
import torch.nn as nn
import torch.nn.functional as f

__all__ = [
    "seed_everything",
    "split_dataframe",
    "AverageMeter",
    "enable_backend_opt",
    "build_optimizer",
    "build_scheduler",
    "FocalLoss",
]


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def enable_backend_opt(cfg):
    """根据 config 打开 cudnn benchmark, TF32, matmul 高精"""
    torch.backends.cudnn.benchmark = True
    if cfg.train.tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    # 2.3 起的新接口：float32 matmul 精度等级
    torch.set_float32_matmul_precision("high")


def split_dataframe(df, valid_ratio=0.15, seed=42):
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    n_val = int(len(df) * valid_ratio)
    train_df = df[n_val:].reset_index(drop=True)
    val_df = df[:n_val].reset_index(drop=True)
    return train_df, val_df


class AverageMeter:
    """计算并存储平均值"""

    def __init__(self):
        self.cnt = None
        self.sum = None
        self.reset()

    def reset(self):
        self.sum = 0.0
        self.cnt = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.cnt += n

    @property
    def avg(self):
        return self.sum / max(self.cnt, 1)


# ───────────────────────  Optimizer / Scheduler 工厂  ──────────────────────

_OPTIMIZER_MAP = {
    "AdamW": optim.AdamW,
    "Adam": optim.Adam,
    "SGD": optim.SGD,
    # 如需更多，按此格式添加…
}

_SCHEDULER_MAP = {
    "cosine": CosineAnnealingLR,
    "step": StepLR,
    "multistep": MultiStepLR,
    "exp": ExponentialLR,
    "plateau": ReduceLROnPlateau,
}


def build_optimizer(model, cfg_train):
    name = cfg_train.optimizer
    if name not in _OPTIMIZER_MAP:
        raise ValueError(f"Unknown optimizer '{name}'")
    kwargs = dict(lr=cfg_train.lr, weight_decay=cfg_train.weight_decay)
    if name == "SGD":
        # 为 SGD 默认加上 momentum + nesterov
        kwargs.update(momentum=0.9, nesterov=True)
    return _OPTIMIZER_MAP[name](model.parameters(), **kwargs)


def build_scheduler(optimizer, cfg_train):
    name = cfg_train.scheduler
    if name not in _SCHEDULER_MAP:
        raise ValueError(f"Unknown scheduler '{name}'")

    # 获取通用参数
    min_lr = getattr(cfg_train, "min_lr", 0.0)
    warmup_epochs = getattr(cfg_train, "warmup_epochs", 0)
    warmup_lr = getattr(cfg_train, "warmup_lr", None)

    # 统一处理 Warmup 逻辑
    if warmup_epochs > 0:
        if name == "plateau":
            raise ValueError(
                "Warmup is not compatible with ReduceLROnPlateau scheduler. "
                "Please set warmup_epochs=0 in the config."
            )

        from torch.optim.lr_scheduler import SequentialLR, LinearLR

        # 计算warmup起始比例
        start_factor = warmup_lr / cfg_train.lr if warmup_lr else 1e-6
        if warmup_lr is None:
            print(f"⚠️ Using default warmup start factor 1e-6 (base_lr * 1e-6 = {cfg_train.lr*1e-6:.1e})")

        # 创建 Warmup 调度器
        warmup = LinearLR(
            optimizer,
            start_factor=start_factor,
            end_factor=1.0,
            total_iters=warmup_epochs,
        )

        # 创建主调度器
        if name == "cosine":
            main_scheduler = CosineAnnealingLR(
                optimizer,
                T_max=cfg_train.epochs - warmup_epochs,
                eta_min=min_lr
            )
        elif name == "step":
            main_scheduler = StepLR(
                optimizer,
                step_size=getattr(cfg_train, "step_size", 10),
                gamma=getattr(cfg_train, "gamma", 0.1),
            )
        elif name == "multistep":
            main_scheduler = MultiStepLR(
                optimizer,
                milestones=getattr(cfg_train, "milestones", [30, 60, 80]),
                gamma=getattr(cfg_train, "gamma", 0.1),
            )
        elif name == "exp":
            main_scheduler = ExponentialLR(
                optimizer,
                gamma=getattr(cfg_train, "gamma", 0.9),
            )
        else:
            raise RuntimeError("Unsupported scheduler")

        return SequentialLR(
            optimizer,
            schedulers=[warmup, main_scheduler],
            milestones=[warmup_epochs],
        )

    else:
        # 无 Warmup 的情况
        if name == "cosine":
            return CosineAnnealingLR(
                optimizer,
                T_max=cfg_train.epochs,
                eta_min=min_lr
            )
        elif name == "step":
            return StepLR(
                optimizer,
                step_size=getattr(cfg_train, "step_size", 10),
                gamma=getattr(cfg_train, "gamma", 0.1),
            )
        elif name == "multistep":
            return MultiStepLR(
                optimizer,
                milestones=getattr(cfg_train, "milestones", [30, 60, 80]),
                gamma=getattr(cfg_train, "gamma", 0.1),
            )
        elif name == "exp":
            return ExponentialLR(
                optimizer,
                gamma=getattr(cfg_train, "gamma", 0.9),
            )
        elif name == "plateau":
            return ReduceLROnPlateau(
                optimizer,
                mode="max",
                factor=getattr(cfg_train, "factor", 0.5),
                patience=getattr(cfg_train, "patience", 3),
                min_lr=min_lr
            )
        else:
            raise ValueError(f"Unsupported scheduler: {name}")


class FocalLoss(nn.Module):
    """
    Focal Loss for multi-class classification.
    gamma: focusing parameter; weight: per-class weights tensor.
    """

    def __init__(self, gamma: float = 2.0, weight: torch.Tensor | None = None):
        super().__init__()
        self.gamma = gamma
        self.weight = weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = f.cross_entropy(logits, targets, weight=self.weight, reduction='none')
        pt = torch.exp(-ce_loss)
        focal = ((1 - pt) ** self.gamma) * ce_loss
        return focal.mean()
