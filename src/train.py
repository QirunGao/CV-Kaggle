import os
import numpy as np
import pandas as pd
import torch

from kornia.augmentation import Normalize as KNormalize
from kornia.augmentation.auto import RandAugment
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchmetrics import MetricCollection
from torchmetrics.classification import (
    MulticlassPrecision,
    MulticlassRecall,
    MulticlassF1Score,
)
from tqdm import tqdm

from src.config import cfg
from src.data.retino_dataset import RetinoDataset
from src.models.build import build_model
from src.utils import FocalLoss
from src.utils import (
    seed_everything,
    split_dataframe,
    AverageMeter,
    enable_backend_opt,
    build_optimizer,
    build_scheduler,
    apply_mixup_cutmix,
)


# ────────────────────────────────────── main ──────────────────────────────────────
def main():
    """
    Main training loop:
    1. Set random seed and enable backend optimizations
    2. Create output directories
    3. Read and split the dataset
    4. Select device
    5. Setup GPU-based RandAugment and Normalize
    6. Set up gradient accumulation
    7. Build DataLoaders
    8. Build model, optimizer, scheduler, and EMA
    9. Training loop
    10. Scheduler step and validation
    11. Save best model for each interval
    12. Save overall best checkpoint
    """
    # ──────────────── 1. Set random seed and enable backend optimizations ────────────────
    seed_everything(cfg.train.seed)
    enable_backend_opt(cfg)

    # ──────────────── 2. Prepare output directory and initialize interval parameters ────────────────
    os.makedirs(cfg.output.dir, exist_ok=True)
    ckpt_dir = os.path.join(cfg.output.dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    # Number of epochs between saving the best interval checkpoint
    save_interval = cfg.output.get("save_interval", 1)
    # Track best F1 and state for the current interval
    interval_best_f1 = 0.0
    interval_best_state = None

    # ──────────────── 3. Read and split the dataset ────────────────
    df = pd.read_csv(cfg.data.train_csv)
    train_df, val_df = split_dataframe(df, cfg.data.valid_ratio, cfg.train.seed)

    print(f"Training dataset size: {len(train_df)}")
    print(f"Validation dataset size: {len(val_df)}")
    print(f"Train class distribution:\n", train_df[cfg.data.col_label].value_counts())
    print(f"Validation class distribution:\n", val_df[cfg.data.col_label].value_counts())

    # ──────────────── 4. Device selection ────────────────
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ──────────────── 5. GPU-side RandAugment and Normalize ────────────────
    rand_aug_gpu = None
    gpu_normalize = None
    if getattr(cfg.train, "rand_aug_gpu", False):
        rand_aug_gpu = RandAugment(n=2, m=9).to(device)
        gpu_normalize = KNormalize(
            mean=torch.tensor((0.485, 0.456, 0.406), device=device),
            std=torch.tensor((0.229, 0.224, 0.225), device=device),
        )

    # ──────────────── 6. Gradient accumulation setup ────────────────
    accum_steps = getattr(cfg.train, "accum_steps", 1)

    # ──────────────── 7. Build DataLoaders ────────────────
    if cfg.train.use_oversampling:
        counts = train_df[cfg.data.col_label].value_counts().to_dict()
        sample_weights = train_df[cfg.data.col_label].map(
            lambda cls: 1.0 / counts[cls]
        ).values
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True
        )
        shuffle = False
    else:
        sampler = None
        shuffle = True

    train_loader = DataLoader(
        RetinoDataset(train_df, train=True),
        sampler=sampler,
        shuffle=shuffle,
        batch_size=cfg.train.batch_size,
        num_workers=cfg.train.num_workers,
        prefetch_factor=cfg.train.prefetch_factor,
        pin_memory=True,
        persistent_workers=True,
    )
    val_loader = DataLoader(
        RetinoDataset(val_df, train=False),
        shuffle=False,
        batch_size=cfg.train.batch_size,
        num_workers=cfg.train.num_workers,
        prefetch_factor=cfg.train.prefetch_factor,
        pin_memory=True,
        persistent_workers=True,
    )

    # ──────────────── 8. Build model, optimizer, scheduler, EMA ────────────────
    model = build_model().to(device, memory_format=torch.channels_last)
    if cfg.train.compile:
        model = torch.compile(model, mode=cfg.train.compile_mode)

    # Layer-wise fine-tuning: freeze backbone parameters for the first `freeze_epochs`
    freeze_epochs = getattr(cfg.train, "freeze_epochs", 0)
    if freeze_epochs > 0:
        print(f"Freezing backbone parameters for the first {freeze_epochs} epochs")
        for name, param in model.named_parameters():
            # Only keep the classification head trainable initially
            if not name.startswith("head."):
                param.requires_grad = False    # Layer-wise fine-tuning: freeze backbone parameters

    # Class‑Balanced Loss
    # Reference: “Class‑Balanced Loss Based on Effective Number of Samples”
    # https://arxiv.org/abs/1901.05555
    if getattr(cfg.train, "use_cb_loss", False):
        # 1) Count the number of samples for each class
        counts = train_df[cfg.data.col_label].value_counts().sort_index().values.astype(np.float32)
        # 2) Set β (typically between 0.9–0.9999), specified in config: cfg.train.cb_beta
        beta = getattr(cfg.train, "cb_beta", 0.9999)
        # 3) Compute effective number of samples: E_n = (1 - βⁿ) / (1 - β)
        effective_num = 1.0 - np.power(beta, counts)
        weights = (1.0 - beta) / (effective_num + 1e-8)
        # 4) Normalize the weights so that ∑w_i = number of classes
        weights = weights / np.sum(weights) * len(weights)
        weight = torch.tensor(weights, dtype=torch.float32, device=device)
        # 5) Construct CB + Focal Loss
        criterion = FocalLoss(
            gamma=getattr(cfg.train, "focal_gamma", 2.0),
            weight=weight
        )
    else:
        # Original FocalLoss (without class weighting)
        criterion = FocalLoss(
            gamma=getattr(cfg.train, "focal_gamma", 2.0),
            weight=None
        )

    optimizer = build_optimizer(model, cfg.train)
    scheduler = build_scheduler(optimizer, cfg.train)
    scaler = GradScaler("cuda", enabled=torch.cuda.is_available())

    from timm.utils import ModelEmaV2
    ema = (
        ModelEmaV2(model, decay=cfg.train.ema_decay)
        if cfg.train.ema_decay
        else None
    )

    best_val_f1 = 0.0
    # List to keep saved interval checkpoint filenames
    interval_files = []

    # ──────────────── 9. Training Loop ────────────────
    for epoch in range(1, cfg.train.epochs + 1):
        if freeze_epochs > 0 and epoch == freeze_epochs + 1:
            print("Unfreezing backbone parameters and rebuilding optimizer/scheduler")
            for param in model.parameters():
                param.requires_grad = True
            optimizer = build_optimizer(model, cfg.train)
            scheduler = build_scheduler(optimizer, cfg.train)
        model.train()
        loss_meter = AverageMeter()
        optimizer.zero_grad(set_to_none=True)

        pbar = tqdm(
            train_loader,
            ncols=100,
            desc=f"Train Epoch {epoch}/{cfg.train.epochs}",
        )

        for step, (imgs, labels) in enumerate(pbar, start=1):
            imgs = imgs.to(device, non_blocking=True)  # type: ignore
            labels = labels.to(device, non_blocking=True)

            # Apply GPU-side augmentation if enabled
            if rand_aug_gpu:
                imgs = rand_aug_gpu(imgs)
                imgs = gpu_normalize(imgs)

            # Handle CutMix / MixUp logic
            imgs, labels_a, labels_b, lam = apply_mixup_cutmix(
                imgs,
                labels,
                cutmix_alpha=cfg.train.cutmix_alpha,
                mixup_alpha=cfg.train.mixup_alpha,
            )

            # Compute loss and backpropagation
            with autocast(device_type='cuda'):
                logits = model(imgs)
                loss = lam * criterion(logits, labels_a) + (1 - lam) * criterion(logits, labels_b)

            scaler.scale(loss).backward()

            # Optimizer step and EMA update with gradient accumulation
            if step % accum_steps == 0 or step == len(train_loader):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                if ema:
                    ema.update(model)

            loss_meter.update(loss.item() * accum_steps, n=imgs.shape[0])
            pbar.set_postfix(loss=loss_meter.avg)

        # ──────────────── 10. Scheduler step & validation metrics ────────────────
        if cfg.train.scheduler == "plateau":
            eval_model = ema.module if ema else model
            val_metrics = _validate(eval_model, val_loader, device)
            scheduler.step(metrics=val_metrics['macro_f1'])
        else:
            scheduler.step()
            eval_model = ema.module if ema else model
            val_metrics = _validate(eval_model, val_loader, device)

        val_macro_f1 = val_metrics['macro_f1'].item()
        print(f"Epoch {epoch} — Macro F1: {val_macro_f1:.4f}")
        for i in range(cfg.model.num_classes):
            p = val_metrics['precision'][i].item()
            r = val_metrics['recall'][i].item()
            f = val_metrics['f1'][i].item()
            print(f"    Class {i:>2} — Prec: {p:.4f}, Rec: {r:.4f}, F1: {f:.4f}")

        # ──────────────── 11. Interval-based checkpoint saving ────────────────
        # Update interval best if current F1 exceeds previous best in this interval
        if val_macro_f1 > interval_best_f1:
            interval_best_f1 = val_macro_f1
            interval_best_state = (ema.module if ema else model).state_dict()

        # At the end of each interval or the final epoch, save the best interval model
        if epoch % save_interval == 0 or epoch == cfg.train.epochs:
            if interval_best_state is not None:
                interval_idx = (epoch - 1) // save_interval + 1
                interval_fname = f"interval_{interval_idx}_best_{interval_best_f1:.4f}.pt"
                torch.save(interval_best_state, os.path.join(ckpt_dir, interval_fname))
                interval_files.append(interval_fname)
            # Reset interval tracking for the next period
            interval_best_f1 = 0.0
            interval_best_state = None

            # ──────────────── 12. Save best-performing checkpoint (by macro F1 score) ────────────────
            if val_macro_f1 > best_val_f1:
                best_val_f1 = val_macro_f1
                fname = f"best_{best_val_f1:.4f}.pt"
                torch.save(
                    (ema.module if ema else model).state_dict(),
                    os.path.join(ckpt_dir, fname),
                )
                # ---- only prune global-best checkpoints, leave interval_*.pt alone ----
                best_files = sorted(f for f in os.listdir(ckpt_dir) if f.startswith("best_"))
                # if there are more than save_top_k global-best files, delete the oldest ones
                while len(best_files) > cfg.output.save_top_k:
                    old = best_files.pop(0)
                    os.remove(os.path.join(ckpt_dir, old))

    print("Training completed.")


# ─────────────────────────────── Validation Function ───────────────────────────────
def _validate(model, loader, device):
    """
    Evaluate the model on the validation set and compute the following metrics:
      - Per-class precision
      - Per-class recall
      - Per-class F1 score
      - Macro-averaged F1 score

    Returns
    -------
    A dictionary containing metric tensors.
    """
    metrics = MetricCollection({
        'precision': MulticlassPrecision(num_classes=cfg.model.num_classes, average=None),
        'recall': MulticlassRecall(num_classes=cfg.model.num_classes, average=None),
        'f1': MulticlassF1Score(num_classes=cfg.model.num_classes, average=None),
        'macro_f1': MulticlassF1Score(num_classes=cfg.model.num_classes, average='macro'),
    }).to(device)

    model.eval()
    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            labels = labels.to(device)
            with autocast(device_type='cuda'):
                logits = model(imgs)
            preds = logits.argmax(dim=1)
            metrics.update(preds, labels)

    return metrics.compute()


if __name__ == "__main__":
    main()
