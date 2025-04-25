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
    "apply_mixup_cutmix",
    "BalancedSoftmaxCE",
]


def rand_bbox(size, lam):
    """
    Generate a random bounding box for CutMix.

    Args:
      size: input tensor dimensions (B, C, H, W)
      lam: mix ratio λ

    Returns:
      bbx1, bby1, bbx2, bby2: top-left and bottom-right coordinates of the crop box
    """
    w, h = size[3], size[2]
    cut_rat = np.sqrt(1.0 - lam)
    cut_w, cut_h = int(w * cut_rat), int(h * cut_rat)
    cx = np.random.randint(w)
    cy = np.random.randint(h)

    bbx1 = np.clip(cx - cut_w // 2, 0, w)
    bby1 = np.clip(cy - cut_h // 2, 0, h)
    bbx2 = np.clip(cx + cut_w // 2, 0, w)
    bby2 = np.clip(cy + cut_h // 2, 0, h)
    return bbx1, bby1, bbx2, bby2


def apply_mixup_cutmix(imgs: torch.Tensor,
                       labels: torch.Tensor,
                       cutmix_alpha: float = 0.0,
                       mixup_alpha: float = 0.0):
    """
    Choose to apply CutMix or MixUp based on probability and alpha values.
    Returns processed imgs, labels_a, labels_b, and λ.
    """
    if cutmix_alpha > 0 and mixup_alpha > 0 and np.random.rand() < 0.5:
        # Apply CutMix
        lam = np.random.beta(cutmix_alpha, cutmix_alpha)
        perm = torch.randperm(imgs.size(0), device=imgs.device)
        bbx1, bby1, bbx2, bby2 = rand_bbox(imgs.size(), lam)
        imgs[:, :, bby1:bby2, bbx1:bbx2] = imgs[perm, :, bby1:bby2, bbx1:bbx2]
        lam = 1 - (bbx2 - bbx1) * (bby2 - bby1) / (imgs.size(-1) * imgs.size(-2))
        labels_a, labels_b = labels, labels[perm]
    elif cutmix_alpha > 0:
        # Apply only CutMix
        lam = np.random.beta(cutmix_alpha, cutmix_alpha)
        perm = torch.randperm(imgs.size(0), device=imgs.device)
        bbx1, bby1, bbx2, bby2 = rand_bbox(imgs.size(), lam)
        imgs[:, :, bby1:bby2, bbx1:bbx2] = imgs[perm, :, bby1:bby2, bbx1:bbx2]
        lam = 1 - (bbx2 - bbx1) * (bby2 - bby1) / (imgs.size(-1) * imgs.size(-2))
        labels_a, labels_b = labels, labels[perm]
    elif mixup_alpha > 0:
        # Apply only MixUp
        lam = np.random.beta(mixup_alpha, mixup_alpha)
        perm = torch.randperm(imgs.size(0), device=imgs.device)
        imgs = lam * imgs + (1 - lam) * imgs[perm]
        labels_a, labels_b = labels, labels[perm]
    else:
        # No mixing applied
        lam = 1.0
        labels_a = labels_b = labels
    return imgs, labels_a, labels_b, lam


def seed_everything(seed: int = 42):
    """
    Fix all random seeds to ensure reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def enable_backend_opt(cfg):
    """
    Enable backend performance optimizations based on configuration:
      - cudnn.benchmark
      - TF32 math precision
      - High-precision mode for float32 matrix multiplications
    """
    torch.backends.cudnn.benchmark = True
    if cfg.train.tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    if torch.cuda.is_available():
        torch.backends.cuda.enable_flash_sdp(True)  # Flash kernel
    torch.backends.cuda.enable_mem_efficient_sdp(True)  # 回退


def split_dataframe(df, valid_ratio: float = 0.15, seed: int = 42):
    """
    Shuffle and split a DataFrame into training and validation sets.

    Args:
      df:          The original DataFrame.
      valid_ratio: Proportion of samples to use for validation.
      seed:        Random seed for shuffling.

    Returns:
      train_df, val_df
    """
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    n_val = int(len(df) * valid_ratio)
    return df[n_val:].reset_index(drop=True), df[:n_val].reset_index(drop=True)


class AverageMeter:
    """
    Computes and stores the running average of values.
    """

    def __init__(self):
        self.count = None
        self.sum = None
        self.reset()

    def reset(self):
        """Reset sum and count."""
        self.sum = 0.0
        self.count = 0

    def update(self, value: float, n: int = 1):
        """
        Add a new value.

        Args:
          value: The new scalar or batch average to add.
          n:     Number of samples that this value represents.
        """
        self.sum += value * n
        self.count += n

    @property
    def avg(self) -> float:
        """Return the current average."""
        return self.sum / max(self.count, 1)


# ───────────── Optimizer and Scheduler Mappings ─────────────
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
    Create an optimizer based on training configuration.

    Args:
      model:      The model whose parameters will be optimized.
      cfg_train:  Configuration containing lr, weight_decay, optimizer name, etc.

    Returns:
      An initialized Optimizer instance.
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
    Create a learning rate scheduler based on training configuration, with optional warmup.

    Args:
      optimizer:  The optimizer to be scheduled.
      cfg_train:  Configuration containing scheduler type, epochs, warmup_epochs, etc.

    Returns:
      A scheduler instance (or a SequentialLR for warmup + main scheduler).
    """
    name = cfg_train.scheduler
    if name not in _SCHEDULER_MAP:
        raise ValueError(f"Unknown scheduler '{name}'")

    min_lr = getattr(cfg_train, "min_lr", 0.0)
    warmup_epochs = getattr(cfg_train, "warmup_epochs", 0)
    warmup_lr = getattr(cfg_train, "warmup_lr", None)

    # Warmup phase
    if warmup_epochs > 0:
        if name == "plateau":
            raise ValueError(
                "Warmup is not supported for ReduceLROnPlateau; set warmup_epochs to 0."
            )
        from torch.optim.lr_scheduler import SequentialLR, LinearLR

        start_factor = (warmup_lr / cfg_train.lr) if warmup_lr else 1e-6
        warmup = LinearLR(
            optimizer,
            start_factor=start_factor,
            end_factor=1.0,
            total_iters=warmup_epochs,
        )

        # Main scheduler after warmup
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

    # No warmup phase
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
    Focal Loss implementation for multiclass classification.

    Args:
      gamma:  Focusing parameter.
      weight: Class weight tensor (optional).
    """

    def __init__(self, gamma: float = 2.0, weight: torch.Tensor | None = None):
        super().__init__()
        self.gamma = gamma
        self.weight = weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute the Focal Loss:
          FL = (1 - p_t)^γ * CE(logits, targets)
        """
        ce_loss = f.cross_entropy(logits, targets, weight=self.weight, reduction="none")
        pt = torch.exp(-ce_loss)
        return ((1 - pt) ** self.gamma * ce_loss).mean()


class BalancedSoftmaxCE(nn.Module):
    """
    Balanced Softmax Cross-Entropy
    (Ren et al., NeurIPS 2020 “BS-CE for Long-Tailed Recognition”)

    logits' = logits + log(n_j)      # n_j: 每类样本数
    loss    = CE(logits', targets)
    """

    def __init__(self, class_counts: torch.Tensor):
        super().__init__()
        # log_prior 形状 [num_classes]
        log_prior = torch.log(class_counts.float().clamp(min=1.0))
        self.register_buffer("log_prior", log_prior)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return f.cross_entropy(logits + self.log_prior, targets)
