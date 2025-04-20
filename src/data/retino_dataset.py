"""
RetinoDataset  ──────────────────────────────────────────────────────────────
• 极速 libjpeg‑turbo 解码：torchvision.io.decode_jpeg
• Ben‑Graham 预处理可选（离线做完就跳过）
• 训练阶段若 cfg.train.rand_aug_gpu=True，不在 Dataset 内做 RandAug
  —— 改由 train.py 在 GPU 完成

2025‑04‑19 更新
──────────────────────────────────────────────────────────────────────────────
* 去掉 `ToTensorV2(dtype=...)`，提高对旧版 Albumentations 的兼容性；
* 统一在 Dataset 中将图像转换为 `float32` 并缩放到 [0,1]；
  若 `ToTensorV2` 已经输出 float 型且范围在 [0,1]，则不会再次除以 255。
"""
from __future__ import annotations

import numpy as np
import torch
import torchvision

import cv2
from albumentations import Compose, Resize, Normalize
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset

from src.config import cfg


def crop_image_from_gray(img: np.ndarray, tol: int = 7) -> np.ndarray:
    """去除四周灰度低于 tol 的黑边。支持单通道/三通道。"""
    if img.ndim == 2:
        mask = img > tol
        return img[np.ix_(mask.any(1), mask.any(0))]

    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    mask = gray > tol
    if not mask.any():
        return img
    img_r = img[:, :, 0][np.ix_(mask.any(1), mask.any(0))]
    img_g = img[:, :, 1][np.ix_(mask.any(1), mask.any(0))]
    img_b = img[:, :, 2][np.ix_(mask.any(1), mask.any(0))]
    return np.stack([img_r, img_g, img_b], axis=-1)


def ben_preprocess(img: np.ndarray, sigma_x: float = 10.0) -> np.ndarray:
    """Ben Graham 对比度增强：4*img – 4*GaussianBlur(img) + 128"""
    img_cropped = crop_image_from_gray(img)
    blurred = cv2.GaussianBlur(img_cropped, (0, 0), sigma_x)
    return cv2.addWeighted(img_cropped, 4, blurred, -4, 128)


def _get_tfms(is_train: bool) -> Compose:
    """返回 Albumentations 变换管线。

    • 始终输出 `torch.Tensor`，首要步骤：Resize → ToTensorV2。
    • 若训练阶段启用了 GPU RandAugment，则 **不在这里** 做 Normalize，
      由 `train.py` 在 GPU 侧完成；
      否则直接插入 Normalize(mean=0.5, std=0.5)。
    """
    sz = cfg.train.img_size
    tfms = [
        Resize(sz, sz),
        ToTensorV2()
    ]
    if not (is_train and cfg.train.rand_aug_gpu):
        tfms.insert(1, Normalize(
            mean=(0.485, 0.456, 0.406),  # 注意括号改成圆括号
            std=(0.229, 0.224, 0.225)
        ))
    return Compose(tfms)


class RetinoDataset(Dataset):
    """
    视网膜 DR 数据集封装。

    Args
    ----
    df    : DataFrame，含 [cfg.data.col_id, cfg.data.col_label]
    train : 是否训练模式
    """

    def __init__(self, df, train: bool = True):
        self.df = df.reset_index(drop=True)
        self.train = train
        self.tfms = _get_tfms(train)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        img_name = row[cfg.data.col_id]
        img_path = f"{cfg.data.img_dir}/{img_name}.jpeg"

        # 1. 极速 JPEG 解码 → HWC, uint8
        img_rgb = torchvision.io.decode_jpeg(
            torchvision.io.read_file(img_path),
            mode=torchvision.io.ImageReadMode.RGB
        ).permute(1, 2, 0).numpy()

        # 2. Ben‑Graham（若未离线预处理）
        if not cfg.data.use_ben_offline:
            sigma = getattr(cfg.data, "ben_sigma", 10.0)
            img_rgb = ben_preprocess(img_rgb, sigma_x=sigma)

        # 3. Albumentations：Resize / (Normalize) / ToTensor
        img_tensor = self.tfms(image=img_rgb)["image"]

        # 4. 确保 float32 & [0,1]，兼容不同版本 Albumentations
        if img_tensor.dtype.is_floating_point:
            if img_tensor.max() > 1.0:
                img_tensor = img_tensor.div(255.0)
        else:
            img_tensor = img_tensor.float().div_(255.0)

        label = torch.tensor(row[cfg.data.col_label], dtype=torch.long)
        return img_tensor, label
