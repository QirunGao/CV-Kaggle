#!/usr/bin/env python3
"""
Compute classification accuracy of a submission file.

Assumptions:
- The prediction file is at output/submission.csv
- The ground-truth labels file is at input/diabetic-retinopathy-resized/trainLabels_cropped.csv
- Both files share the same key column (usually called "image" or "id")
- The ground-truth label column is named "level", and the prediction column in submission.csv is named "diagnosis"
If your column names differ, modify the constants below accordingly.
"""

import pandas as pd
from pathlib import Path

# === Editable constants ===
PRED_PATH = Path("output/submission.csv")  # Path to your prediction CSV
GT_PATH = Path("input/diabetic-retinopathy-resized/trainLabels_cropped.csv")  # Path to ground-truth CSV
KEY_COL = "image"  # Name of the key column in both files
GT_LABEL_COL = "level"  # Name of the ground-truth label column
PRED_LABEL_COL = "diagnosis"  # Name of the prediction column in your submission.csv
# ==========================


def main():
    # Load CSV files
    pred_df = pd.read_csv(PRED_PATH)
    gt_df = pd.read_csv(GT_PATH)

    # Verify required columns exist
    for df, path in [(pred_df, PRED_PATH), (gt_df, GT_PATH)]:
        required = {KEY_COL, GT_LABEL_COL if df is gt_df else PRED_LABEL_COL}
        missing = required - set(df.columns)
        if missing:
            raise KeyError(
                f"File {path} is missing columns {missing}; actual columns: {list(df.columns)}"
            )

    # Rename the prediction column to match the ground-truth label name
    pred_df = pred_df.rename(columns={PRED_LABEL_COL: GT_LABEL_COL})

    # Merge on the key column to align true and predicted labels
    merged = pd.merge(
        gt_df[[KEY_COL, GT_LABEL_COL]],
        pred_df[[KEY_COL, GT_LABEL_COL]],
        on=KEY_COL,
        suffixes=("_true", "_pred"),
        how="inner"
    )

    # Calculate accuracy
    correct = merged[f"{GT_LABEL_COL}_true"].eq(merged[f"{GT_LABEL_COL}_pred"])
    accuracy: float = correct.mean()

    print(f"Accuracy: {accuracy:.4%}  ({merged.shape[0]} samples compared)")


if __name__ == "__main__":
    main()
