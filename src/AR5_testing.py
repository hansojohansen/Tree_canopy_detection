#!/usr/bin/env python
"""Compare AR5 raster tiles to ground‑truth annotations.

The script iterates over matching TIFF pairs (annotation vs. AR5 prediction),
computes block‑wise confusion matrices to control memory, and exports per‑file
and global metrics + heat‑maps.

"""

from __future__ import annotations
import os
import argparse
from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import Window
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Evaluate AR5 tree‑cover rasters against annotations.")
parser.add_argument("--anno_dir", type=str, default="./data/testing/annotations", help="Folder with ground‑truth annotation TIFFs")
parser.add_argument("--ar5_dir", type=str, default="./data/AR5", help="Folder with AR5 raster TIFFs")
parser.add_argument("--out_dir", type=str, default="outputs/ar5_metrics", help="Output root directory")
parser.add_argument("--block_size", type=int, default=512, help="Sliding window size (pixels)")
args = parser.parse_args()

ANNO_DIR = Path(args.anno_dir)
AR5_DIR = Path(args.ar5_dir)
OUT_DIR = Path(args.out_dir)
OUT_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------------------------------------------------------------
# helper
# -----------------------------------------------------------------------------

def process_blocks(anno_tif: Path, ar5_tif: Path, blk: int) -> np.ndarray:
    """Return 2×2 confusion matrix aggregated over blocks."""
    with rasterio.open(anno_tif) as src_t, rasterio.open(ar5_tif) as src_p:
        if (src_t.width, src_t.height) != (src_p.width, src_p.height):
            raise ValueError(f"Mismatched dimensions: {anno_tif} vs {ar5_tif}")
        cm = np.zeros((2, 2), dtype=np.int64)
        for y in range(0, src_t.height, blk):
            h = min(blk, src_t.height - y)
            for x in range(0, src_t.width, blk):
                w = min(blk, src_t.width - x)
                win = Window(x, y, w, h)
                true = src_t.read(1, window=win)
                pred = src_p.read(1, window=win)

                mask = np.ones(true.shape, dtype=bool)
                if src_t.nodata is not None:
                    mask &= true != src_t.nodata
                if src_p.nodata is not None:
                    mask &= pred != src_p.nodata
                if not np.any(mask):
                    continue
                cm += confusion_matrix(true[mask].astype(np.uint8).flatten(), pred[mask].astype(np.uint8).flatten(), labels=[0, 1])
        return cm

# -----------------------------------------------------------------------------
# gather matching files
# -----------------------------------------------------------------------------
anno_files = {p.name: p for p in ANNO_DIR.glob("*.tif")}
ar5_files = {p.name: p for p in AR5_DIR.glob("*.tif")}
common = sorted(set(anno_files).intersection(ar5_files))
if not common:
    raise ValueError("No matching TIFF names between annotations and AR5 directory.")

combined_cm = np.zeros((2, 2), dtype=np.int64)

# -----------------------------------------------------------------------------
# per‑file evaluation
# -----------------------------------------------------------------------------
for name in common:
    cm = process_blocks(anno_files[name], ar5_files[name], args.block_size)
    combined_cm += cm

    # metrics
    tp, tn, fp, fn = cm[1, 1], cm[0, 0], cm[0, 1], cm[1, 0]
    acc = (tp + tn) / cm.sum()
    prec = tp / (tp + fp + 1e-8)
    rec = tp / (tp + fn + 1e-8)
    f1 = 2 * prec * rec / (prec + rec + 1e-8)

    out_dir = OUT_DIR / Path(name).stem
    out_dir.mkdir(exist_ok=True)

    # save heat‑map
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm_norm, annot=True, fmt=".2%", cmap="Blues", xticklabels=["Pred Other", "Pred Tree"], yticklabels=["Actual Other", "Actual Tree"])
    plt.title(f"Confusion Matrix – {name}")
    plt.tight_layout()
    plt.savefig(out_dir / "confusion_matrix.png")
    plt.close()

    # save metrics txt
    with open(out_dir / "metrics.txt", "w") as f:
        f.write(f"Accuracy : {acc*100:.2f}%\n")
        f.write(f"Precision: {prec*100:.2f}%\n")
        f.write(f"Recall   : {rec*100:.2f}%\n")
        f.write(f"F1 Score : {f1*100:.2f}%\n\n")
        f.write("Confusion Matrix (counts):\n")
        f.write(np.array2string(cm, separator=", "))

    print(f"[INFO] Saved metrics for {name}")

# -----------------------------------------------------------------------------
# combined metrics
# -----------------------------------------------------------------------------
TP, TN, FP, FN = combined_cm[1, 1], combined_cm[0, 0], combined_cm[0, 1], combined_cm[1, 0]
acc_c = (TP + TN) / combined_cm.sum()
prec_c = TP / (TP + FP + 1e-8)
rec_c = TP / (TP + FN + 1e-8)
f1_c = 2 * prec_c * rec_c / (prec_c + rec_c + 1e-8)

cm_norm_c = combined_cm.astype(float) / combined_cm.sum(axis=1, keepdims=True)
plt.figure(figsize=(6, 5))
sns.heatmap(cm_norm_c, annot=True, fmt=".2%", cmap="Blues", xticklabels=["Pred Other", "Pred Tree"], yticklabels=["Actual Other", "Actual Tree"])
plt.title("Combined Confusion Matrix")
plt.tight_layout()
plt.savefig(OUT_DIR / "combined_confusion_matrix.png")
plt.close()

with open(OUT_DIR / "combined_metrics.txt", "w") as f:
    f.write(f"Accuracy : {acc_c*100:.2f}%\n")
    f.write(f"Precision: {prec_c*100:.2f}%\n")
    f.write(f"Recall   : {rec_c*100:.2f}%\n")
    f.write(f"F1 Score : {f1_c*100:.2f}%\n\n")
    f.write("Confusion Matrix (counts):\n")
    f.write(np.array2string(combined_cm, separator=", "))

print("[INFO] Combined metrics saved →", OUT_DIR)
