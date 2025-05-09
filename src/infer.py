from __future__ import annotations

import os
import pandas as pd
import torch

from torch.utils.data import DataLoader
from tqdm import tqdm

from src.config import cfg
from src.data.retino_dataset import RetinoDataset
from src.models.build import build_model
from src.utils import enable_backend_opt


def main():
    # ─────────────────────────────
    # 1. Enable backend optimizations (e.g., cudnn benchmark)
    # ─────────────────────────────
    enable_backend_opt(cfg)

    # ─────────────────────────────
    # 2. Automatically load the latest checkpoint
    # ─────────────────────────────
    ckpt_dir = os.path.join(cfg.output.dir, "checkpoints")
    ckpts = sorted(os.listdir(ckpt_dir))
    if not ckpts:
        raise FileNotFoundError(f"No checkpoints in {ckpt_dir}")
    ckpt_path = os.path.join(ckpt_dir, ckpts[-1])

    # ─────────────────────────────
    # 3. Load test data (by default, use the first 100 samples as a demo)
    # ─────────────────────────────
    df = pd.read_csv(cfg.data.train_csv).iloc[:100].copy()
    dl_kwargs = dict(
        batch_size=cfg.train.batch_size,
        num_workers=cfg.train.num_workers,
        prefetch_factor=cfg.train.prefetch_factor,
        pin_memory=True,
        persistent_workers=True,
    )
    test_loader = DataLoader(
        RetinoDataset(df, train=False),
        shuffle=False,
        **dl_kwargs
    )

    # ─────────────────────────────
    # 4. Build the model & load weights
    # ─────────────────────────────
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model().to(device, memory_format=torch.channels_last)

    if cfg.train.compile:
        model = torch.compile(model, mode=cfg.train.compile_mode)

    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()

    # ─────────────────────────────
    # 5. Run inference
    # ─────────────────────────────
    all_preds = []
    with torch.no_grad(), torch.cuda.amp.autocast():
        for imgs, _ in tqdm(test_loader, ncols=100, desc="Infer"):
            imgs = imgs.to(device, non_blocking=True)
            logits = model(imgs)
            preds = logits.softmax(dim=1).argmax(dim=1)
            all_preds.extend(preds.cpu().tolist())

    # ─────────────────────────────
    # 6. Save inference results to CSV
    # ─────────────────────────────
    out_df = pd.DataFrame({
        cfg.data.col_id: df[cfg.data.col_id],
        "diagnosis": all_preds
    })
    os.makedirs(cfg.output.dir, exist_ok=True)
    out_path = os.path.join(cfg.output.dir, "submission.csv")
    out_df.to_csv(out_path, index=False)
    print(f"Saved submission to {out_path}")


if __name__ == "__main__":
    main()
