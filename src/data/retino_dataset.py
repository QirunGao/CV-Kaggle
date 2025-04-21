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
    """
    从图像中裁剪掉灰度值低于 tol 的边缘区域。
    适用于单通道（灰度）或三通道（RGB）图像。
    """
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
    """
    应用 Ben Graham 风格的图像预处理，用于增强视网膜图像的对比度。
    操作：增强图像细节，执行：4*原图 - 4*模糊图 + 128
    """
    img_cropped = crop_image_from_gray(img)
    blurred = cv2.GaussianBlur(img_cropped, (0, 0), sigma_x)
    return cv2.addWeighted(img_cropped, 4, blurred, -4, 128)


def _get_tfms(is_train: bool) -> Compose:
    """
    构造 Albumentations 图像增强管线（变换组合器）。

    - Resize 是所有阶段的统一步骤；
    - 若使用 GPU 端 RandAugment，则仅缩放图像至 [0,1]，不做标准化；
    - 否则执行完整 Normalize(mean/std) 和 ToTensorV2 转换。
    """
    sz = cfg.train.img_size
    if is_train and cfg.train.rand_aug_gpu:
        return Compose([
            Resize(sz, sz),
            Normalize(mean=(0.0, 0.0, 0.0),
                      std=(1.0, 1.0, 1.0),
                      max_pixel_value=255.0),
            ToTensorV2(),
        ])
    else:
        return Compose([
            Resize(sz, sz),
            Normalize(mean=(0.485, 0.456, 0.406),
                      std=(0.229, 0.224, 0.225),
                      max_pixel_value=255.0),
            ToTensorV2(),
        ])


class RetinoDataset(Dataset):
    """
    视网膜图像数据集封装类，用于 DR（糖尿病视网膜病变）分类任务。

    参数
    ----
    df    : 包含图像 ID 和标签的 Pandas DataFrame；
    train : 指示是否为训练模式（控制数据增强策略）。
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

        # 1. 使用 libjpeg-turbo 快速读取 JPEG 图像，格式为 [H, W, C], uint8
        img_rgb = torchvision.io.decode_jpeg(
            torchvision.io.read_file(img_path),
            mode=torchvision.io.ImageReadMode.RGB
        ).permute(1, 2, 0).numpy()

        # 2. 若未使用离线 Ben-Graham 预处理，则在线执行增强
        if not cfg.data.use_ben_offline:
            sigma = getattr(cfg.data, "ben_sigma", 10.0)
            img_rgb = ben_preprocess(img_rgb, sigma_x=sigma)

        # 3. 使用 Albumentations 执行 Resize、Normalize、ToTensor
        img_tensor = self.tfms(image=img_rgb)["image"]

        # 4. 提取图像标签
        label = torch.tensor(row[cfg.data.col_label], dtype=torch.long)
        return img_tensor, label
