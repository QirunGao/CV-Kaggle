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
    Crop out low-intensity regions (below `tol`) from the edges of the image.
    Works for both single-channel (grayscale) and 3-channel (RGB) images.
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
    Apply Ben Graham-style preprocessing to enhance contrast in retinal images.
    Operation: enhance details by 4 * original - 4 * blurred + 128.
    """
    img_cropped = crop_image_from_gray(img)
    blurred = cv2.GaussianBlur(img_cropped, (0, 0), sigma_x)
    return cv2.addWeighted(img_cropped, 4, blurred, -4, 128)


def _get_tfms(is_train: bool) -> Compose:
    """
    Build an Albumentations image transformation pipeline.

    - Resize is applied in all cases.
    - If GPU-based RandAugment is used, normalize only to [0, 1] without standardization.
    - Otherwise, apply full normalization (mean/std) and convert to tensor.
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
    Dataset wrapper for retinal images used in DR (diabetic retinopathy) classification.

    Parameters
    ----------
    df    : A Pandas DataFrame containing image IDs and labels.
    train : Whether to use training mode (controls augmentation behavior).
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

        # 1. Use libjpeg-turbo to quickly load the JPEG image in [H, W, C], uint8 format
        img_rgb = torchvision.io.decode_jpeg(
            torchvision.io.read_file(img_path),
            mode=torchvision.io.ImageReadMode.RGB
        ).permute(1, 2, 0).numpy()

        # 2. Apply Ben Graham preprocessing on-the-fly if offline version is not used
        if not cfg.data.use_ben_offline:
            sigma = getattr(cfg.data, "ben_sigma", 10.0)
            img_rgb = ben_preprocess(img_rgb, sigma_x=sigma)

        # 3. Apply Albumentations: Resize, Normalize, ToTensor
        img_tensor = self.tfms(image=img_rgb)["image"]

        # 4. Extract image label
        label = torch.tensor(row[cfg.data.col_label], dtype=torch.long)
        return img_tensor, label
