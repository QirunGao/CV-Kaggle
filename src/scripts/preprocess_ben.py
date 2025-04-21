#!/usr/bin/env python3
import argparse
import cv2
import glob
import multiprocessing as mp
import os
from functools import partial
from typing import Sequence

from tqdm import tqdm

from src.data.retino_dataset import ben_preprocess


def _worker(out_dir: str, sigma_x: float, path: str) -> None:
    fn = os.path.basename(path)
    # Read and convert to RGB
    img_bgr = cv2.imread(path)
    img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    # Apply preprocessing
    img = ben_preprocess(img, sigma_x)
    # Use tuple as Sequence[int] for `params` to avoid type-checking warnings
    params: Sequence[int] = (cv2.IMWRITE_JPEG_QUALITY, 95)
    # Write the image (converted back to BGR)
    success = cv2.imwrite(
        os.path.join(out_dir, fn),
        cv2.cvtColor(img, cv2.COLOR_RGB2BGR),
        params
    )
    if not success:
        raise IOError(f"Failed to write image to {os.path.join(out_dir, fn)}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Preprocess images with ben_preprocess"
    )
    ap.add_argument(
        "--src_dir", required=True,
        help="Directory containing source .jpeg images"
    )
    ap.add_argument(
        "--dst_dir", required=True,
        help="Directory to save processed images"
    )
    ap.add_argument(
        "--sigma_x", type=float, default=10.0,
        help="Sigma value for Gaussian blur in ben_preprocess"
    )
    ap.add_argument(
        "--workers", type=int, default=mp.cpu_count(),
        help="Number of parallel worker processes"
    )
    args = ap.parse_args()

    os.makedirs(args.dst_dir, exist_ok=True)
    paths = glob.glob(os.path.join(args.src_dir, "*.jpeg"))
    with mp.Pool(args.workers) as pool:
        for _ in tqdm(
                pool.imap_unordered(
                    partial(_worker, args.dst_dir, args.sigma_x), paths
                ),
                total=len(paths), ncols=100
        ):
            pass


if __name__ == "__main__":
    main()
