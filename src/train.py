import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler

import pandas as pd
from tqdm import tqdm
from kornia.augmentation.auto import RandAugment
from kornia.augmentation import Normalize as KNormalize

from src.config import cfg
from src.data.retino_dataset import RetinoDataset
from src.models.build import build_model
from src.utils import (
    seed_everything,
    split_dataframe,
    AverageMeter,
    enable_backend_opt,
    build_optimizer,
    build_scheduler,
)


def rand_bbox(size, lam):
    """
    Generate a random bounding box for CutMix.
    Args:
        size: input tensor size (B, C, H, W)
        lam: mixing ratio
    Returns:
        coordinates (bbx1, bby1, bbx2, bby2)
    """
    w, h = size[3], size[2]
    cut_rat = np.sqrt(1.0 - lam)  # Keep the same for bounding box size.
    cut_w, cut_h = int(w * cut_rat), int(h * cut_rat)
    cx = np.random.randint(w)
    cy = np.random.randint(h)

    bbx1 = np.clip(cx - cut_w // 2, 0, w)
    bby1 = np.clip(cy - cut_h // 2, 0, h)
    bbx2 = np.clip(cx + cut_w // 2, 0, w)
    bby2 = np.clip(cy + cut_h // 2, 0, h)
    return bbx1, bby1, bbx2, bby2


# ─────────────────────────────────── main ──────────────────────────────────
def main():
    # 1. 随机种子与后端优化
    seed_everything(cfg.train.seed)
    enable_backend_opt(cfg)

    # 2. 输出目录
    os.makedirs(cfg.output.dir, exist_ok=True)
    ckpt_dir = os.path.join(cfg.output.dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    # 3. 读取数据并划分
    df = pd.read_csv(cfg.data.train_csv)
    train_df, val_df = split_dataframe(df, cfg.data.valid_ratio, cfg.train.seed)

    print(f"Training dataset size: {len(train_df)}")
    print(f"Validation dataset size: {len(val_df)}")

    # 4. 设备设置
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 5. GPU RandAugment + Normalize (保持与验证集一致)
    rand_aug_gpu = None
    gpu_normalize = None
    if getattr(cfg.train, "rand_aug_gpu", False):
        rand_aug_gpu = RandAugment(n=2, m=9).to(device)
        gpu_normalize = KNormalize(
            mean=torch.tensor((0.485, 0.456, 0.406), device=device),  # 元组
            std=torch.tensor((0.229, 0.224, 0.225), device=device),   # 元组
        )

    # 6. 梯度累积步数
    accum_steps = getattr(cfg.train, "accum_steps", 1)

    # 7. DataLoader
    dl_kwargs = dict(
        batch_size=cfg.train.batch_size,
        num_workers=cfg.train.num_workers,
        prefetch_factor=cfg.train.prefetch_factor,
        pin_memory=True,
        persistent_workers=True,
    )
    train_loader = DataLoader(
        RetinoDataset(train_df, train=True), shuffle=True, **dl_kwargs
    )
    val_loader = DataLoader(
        RetinoDataset(val_df, train=False), shuffle=False, **dl_kwargs
    )

    # 8. 构建模型、优化器、调度器
    model = build_model().to(device, memory_format=torch.channels_last)
    if cfg.train.compile:
        model = torch.compile(model, mode=cfg.train.compile_mode)

    criterion = nn.CrossEntropyLoss(label_smoothing=cfg.train.label_smoothing)
    optimizer = build_optimizer(model, cfg.train)
    scheduler = build_scheduler(optimizer, cfg.train)
    scaler = GradScaler(enabled=torch.cuda.is_available())

    from timm.utils import ModelEmaV2

    ema = (
        ModelEmaV2(model, decay=cfg.train.ema_decay)
        if cfg.train.ema_decay
        else None
    )

    best_val_f1 = 0.0

    # 9. 训练循环
    for epoch in range(1, cfg.train.epochs + 1):
        model.train()
        loss_meter = AverageMeter()
        optimizer.zero_grad(set_to_none=True)

        pbar = tqdm(
            train_loader,
            ncols=100,
            desc=f"Train Epoch {epoch}/{cfg.train.epochs}",
        )

        for step, (imgs, labels) in enumerate(pbar, start=1):
            imgs: torch.Tensor  # type hint
            labels: torch.Tensor

            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            # GPU RandAug + Normalize
            if rand_aug_gpu is not None:
                imgs = rand_aug_gpu(imgs)
                imgs = gpu_normalize(imgs)

            if cfg.train.cutmix_alpha > 0 and cfg.train.mixup_alpha > 0:
                # 随机选择 CutMix 或 Mixup
                if np.random.rand() < 0.5:
                    # 应用 CutMix
                    lam = np.random.beta(cfg.train.cutmix_alpha, cfg.train.cutmix_alpha)
                    perm = torch.randperm(imgs.size(0), device=device)
                    bbx1, bby1, bbx2, bby2 = rand_bbox(imgs.size(), lam)
                    imgs[:, :, bby1:bby2, bbx1:bbx2] = imgs[perm, :, bby1:bby2, bbx1:bbx2]
                    lam = 1 - (bbx2 - bbx1) * (bby2 - bby1) / (imgs.size(-1) * imgs.size(-2))
                    labels_a, labels_b = labels, labels[perm]
                else:
                    # 应用 Mixup
                    lam = np.random.beta(cfg.train.mixup_alpha, cfg.train.mixup_alpha)
                    perm = torch.randperm(imgs.size(0), device=device)
                    imgs = lam * imgs + (1 - lam) * imgs[perm]
                    labels_a, labels_b = labels, labels[perm]

            elif cfg.train.cutmix_alpha > 0:
                # CutMix处理
                lam = np.random.beta(cfg.train.cutmix_alpha, cfg.train.cutmix_alpha)
                perm = torch.randperm(imgs.size(0), device=device)
                bbx1, bby1, bbx2, bby2 = rand_bbox(imgs.size(), lam)
                imgs[:, :, bby1:bby2, bbx1:bbx2] = imgs[perm, :, bby1:bby2, bbx1:bbx2]
                lam = 1 - (bbx2 - bbx1) * (bby2 - bby1) / (imgs.size(-1) * imgs.size(-2))  # 修正lambda计算
                labels_a, labels_b = labels, labels[perm]

            elif cfg.train.mixup_alpha > 0:  # 修正条件为mixup_alpha
                # MixUp处理
                lam = np.random.beta(cfg.train.mixup_alpha, cfg.train.mixup_alpha)
                perm = torch.randperm(imgs.size(0), device=device)
                imgs = lam * imgs + (1 - lam) * imgs[perm]
                labels_a, labels_b = labels, labels[perm]

            else:
                # 无混合增强
                labels_a = labels
                labels_b = labels
                lam = 1.0

            # 统一损失计算（无论是否使用增强）
            with autocast():
                logits = model(imgs)
                loss = lam * criterion(logits, labels_a) + (1 - lam) * criterion(logits, labels_b)

            scaler.scale(loss).backward()

            # 梯度累积：累满步数或最后一步才更新
            if step % accum_steps == 0 or step == len(train_loader):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

                if ema:
                    ema.update(model)

            # 记录真实 loss
            loss_meter.update(loss.item() * accum_steps, n=imgs.shape[0])
            pbar.set_postfix(loss=loss_meter.avg)

        # 10. 调度器 & 验证
        if cfg.train.scheduler == "plateau":
            eval_model = ema.module if ema else model
            val_f1 = _validate(eval_model, val_loader, device)
            scheduler.step(metrics=val_f1)
        else:
            scheduler.step()
            eval_model = ema.module if ema else model
            val_f1 = _validate(eval_model, val_loader, device)

        print(f"Epoch {epoch} — val F1: {val_f1:.4f}")

        # 11. 保存最优
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            fname = f"best_{best_val_f1:.4f}.pt"
            torch.save(
                (ema.module if ema else model).state_dict(),
                os.path.join(ckpt_dir, fname),
            )
            files = sorted(os.listdir(ckpt_dir))
            while len(files) > cfg.output.save_top_k:
                os.remove(os.path.join(ckpt_dir, files.pop(0)))

    print("Training completed.")


# ───────────────────────────────── 验证函数 ───────────────────────────────
def _validate(model, loader, device):
    from torchmetrics.classification import MulticlassF1Score

    metric = MulticlassF1Score(
        num_classes=cfg.model.num_classes, average="macro"
    ).to(device)
    model.eval()  # 进入评估模式
    with torch.no_grad(), autocast():
        for imgs, labels in loader:
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            preds = model(imgs).softmax(dim=1)
            pred_labels = preds.argmax(dim=1)  # 获取预测类别索引
            metric.update(pred_labels, labels)

    return metric.compute().item()


if __name__ == "__main__":
    main()
