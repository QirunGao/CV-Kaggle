import random
import numpy as np
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import (
    CosineAnnealingLR, StepLR, MultiStepLR,
    ExponentialLR, ReduceLROnPlateau
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


def seed_everything(seed: int = 42):
    """
    固定所有随机种子，确保实验可复现。
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def enable_backend_opt(cfg):
    """
    根据配置启用后端性能优化：
      - cudnn.benchmark
      - TF32 运算精度
      - float32 矩阵乘法高精度模式
    """
    torch.backends.cudnn.benchmark = True
    if cfg.train.tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")


def split_dataframe(df, valid_ratio: float = 0.15, seed: int = 42):
    """
    打乱并拆分 DataFrame 为训练集和验证集。

    参数:
      df:          原始 DataFrame
      valid_ratio: 验证集占比
      seed:        随机种子

    返回:
      train_df, val_df
    """
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    n_val = int(len(df) * valid_ratio)
    return df[n_val:].reset_index(drop=True), df[:n_val].reset_index(drop=True)


class AverageMeter:
    """
    计算并跟踪累积的平均值。
    """

    def __init__(self):
        self.count = None
        self.sum = None
        self.reset()

    def reset(self):
        """重置计数和累加和。"""
        self.sum = 0.0
        self.count = 0

    def update(self, value: float, n: int = 1):
        """
        累加新值。

        参数:
          value: 新增的标量或批次平均值
          n:     样本数量
        """
        self.sum += value * n
        self.count += n

    @property
    def avg(self) -> float:
        """返回当前的平均值。"""
        return self.sum / max(self.count, 1)


# ───────────── Optimizer 与 Scheduler 配置映射 ─────────────
_OPTIMIZER_MAP = {
    "AdamW": optim.AdamW,
    "Adam": optim.Adam,
    "SGD": optim.SGD,
}

_SCHEDULER_MAP = {
    "cosine": CosineAnnealingLR,
    "step": StepLR,
    "multistep": MultiStepLR,
    "exp": ExponentialLR,
    "plateau": ReduceLROnPlateau,
}


def build_optimizer(model: nn.Module, cfg_train) -> optim.Optimizer:
    """
    根据训练配置创建优化器。

    参数:
      model:      需优化参数的模型
      cfg_train:  包含 lr、weight_decay、optimizer 名称等

    返回:
      初始化好的 Optimizer 实例
    """
    name = cfg_train.optimizer
    if name not in _OPTIMIZER_MAP:
        raise ValueError(f"Unknown optimizer '{name}'")
    kwargs = dict(lr=cfg_train.lr, weight_decay=cfg_train.weight_decay)
    if name == "SGD":
        kwargs.update(momentum=0.9, nesterov=True)
    return _OPTIMIZER_MAP[name](model.parameters(), **kwargs)


def build_scheduler(optimizer: optim.Optimizer, cfg_train):
    """
    根据训练配置创建学习率调度器，支持可选 warmup。

    参数:
      optimizer:  已创建的优化器
      cfg_train:  包含 scheduler 类型、epochs、warmup_epochs 等

    返回:
      调度器实例（或 SequentialLR）
    """
    name = cfg_train.scheduler
    if name not in _SCHEDULER_MAP:
        raise ValueError(f"Unknown scheduler '{name}'")

    min_lr = getattr(cfg_train, "min_lr", 0.0)
    warmup_epochs = getattr(cfg_train, "warmup_epochs", 0)
    warmup_lr = getattr(cfg_train, "warmup_lr", None)

    # Warmup 阶段
    if warmup_epochs > 0:
        if name == "plateau":
            raise ValueError(
                "Warmup 不适用于 ReduceLROnPlateau 调度器，请将 warmup_epochs 设为 0。"
            )
        from torch.optim.lr_scheduler import SequentialLR, LinearLR

        start_factor = (warmup_lr / cfg_train.lr) if warmup_lr else 1e-6
        warmup = LinearLR(
            optimizer,
            start_factor=start_factor,
            end_factor=1.0,
            total_iters=warmup_epochs,
        )

        # 主调度器
        if name == "cosine":
            main = CosineAnnealingLR(
                optimizer,
                T_max=cfg_train.epochs - warmup_epochs,
                eta_min=min_lr,
            )
        elif name == "step":
            main = StepLR(
                optimizer,
                step_size=getattr(cfg_train, "step_size", 10),
                gamma=getattr(cfg_train, "gamma", 0.1),
            )
        elif name == "multistep":
            main = MultiStepLR(
                optimizer,
                milestones=getattr(cfg_train, "milestones", [30, 60, 80]),
                gamma=getattr(cfg_train, "gamma", 0.1),
            )
        elif name == "exp":
            main = ExponentialLR(
                optimizer,
                gamma=getattr(cfg_train, "gamma", 0.9),
            )
        else:
            raise RuntimeError(f"Unsupported scheduler '{name}'")

        return SequentialLR(
            optimizer,
            schedulers=[warmup, main],
            milestones=[warmup_epochs],
        )

    # 无 Warmup
    if name == "cosine":
        return CosineAnnealingLR(
            optimizer,
            T_max=cfg_train.epochs,
            eta_min=min_lr,
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
    else:  # plateau
        return ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=getattr(cfg_train, "factor", 0.5),
            patience=getattr(cfg_train, "patience", 3),
            min_lr=min_lr,
        )


class FocalLoss(nn.Module):
    """
    Focal Loss 多分类实现。

    参数:
      gamma:  聚焦系数
      weight: 类别权重张量（可选）
    """

    def __init__(self, gamma: float = 2.0, weight: torch.Tensor | None = None):
        super().__init__()
        self.gamma = gamma
        self.weight = weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        计算 Focal Loss：
          FL = (1 - p_t)^γ * CE(logits, targets)
        """
        ce_loss = f.cross_entropy(logits, targets, weight=self.weight, reduction="none")
        pt = torch.exp(-ce_loss)
        return ((1 - pt) ** self.gamma * ce_loss).mean()
