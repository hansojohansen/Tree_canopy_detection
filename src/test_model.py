#!/usr/bin/env python
"""Model evaluation script

Runs inference on the *testing* split, computes a global confusion matrix,
reconstructs full‑image masks and saves them to `outputs/predicted_masks/`.
"""

import os
import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import rasterio
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

from dataloader import segDataset
from model import UNet

# -----------------------------------------------------------------------------
# device
# -----------------------------------------------------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
parser = argparse.ArgumentParser(
    description="Run inference on the testing set, compute confusion matrices and write georeferenced masks.")
parser.add_argument("--data_root", type=str, default="./data", help="Root folder with a 'testing' subdir.")
parser.add_argument("--model_path", type=str, default="./saved_models/best_model.pth", help="Trained model .pth file")
parser.add_argument("--batch_size", type=int, default=4, help="Batch size for DataLoader")
parser.add_argument("--patch_size", type=int, default=124, help="Tile size in pixels")
parser.add_argument("--threshold", type=float, default=0.5, help="Class‑1 probability threshold")
parser.add_argument("--save_dir", type=str, default="outputs/predicted_masks", help="Output root directory")
args = parser.parse_args()

# -----------------------------------------------------------------------------
# paths
# -----------------------------------------------------------------------------
TEST_ROOT = Path(args.data_root) / "testing"
MODEL_PATH = Path(args.model_path)
MODEL_NAME = MODEL_PATH.parent.name or "model"
OUTPUT_DIR = Path(args.save_dir) / MODEL_NAME
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
print(f"Testing data root: {TEST_ROOT}")

# -----------------------------------------------------------------------------
# dataset & loader
# -----------------------------------------------------------------------------
dataset = segDataset(root=str(TEST_ROOT), patch_size=args.patch_size, mode="train", transform=transforms.Compose([]))

def collate(batch):
    if not batch:
        return None
    images, annotations, positions, img_idx = zip(*batch)
    return torch.stack(images), torch.stack(annotations), list(positions), list(img_idx)

loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=collate)

# -----------------------------------------------------------------------------
# model
# -----------------------------------------------------------------------------
model = UNet(n_channels=3, n_classes=2, bilinear=True).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

# -----------------------------------------------------------------------------
# files to accumulate labels (kept to avoid large RAM usage)
# -----------------------------------------------------------------------------
true_file = OUTPUT_DIR / "true_labels.npy"
pred_file = OUTPUT_DIR / "predicted_labels.npy"
for f in (true_file, pred_file):
    if f.exists():
        f.unlink()

predictions_per_image = {}

# -----------------------------------------------------------------------------
# inference loop
# -----------------------------------------------------------------------------
with torch.no_grad():
    for batch in loader:
        if batch is None:
            continue
        imgs, annos, pos, idxs = batch
        imgs = imgs.to(DEVICE)
        probs = torch.softmax(model(imgs), dim=1)[:, 1, :, :]
        preds = (probs > args.threshold).long().cpu().numpy()
        annos_np = annos.cpu().numpy()

        for i in range(imgs.size(0)):
            pred_patch = preds[i]
            true_patch = annos_np[i]
            x, y = pos[i]
            img_idx = idxs[i]

            # append flat arrays to disk to save RAM
            with open(true_file, "ab") as f_t, open(pred_file, "ab") as f_p:
                np.save(f_t, true_patch.flatten())
                np.save(f_p, pred_patch.flatten())

            # stitch mask back together
            if img_idx not in predictions_per_image:
                h, w = dataset.image_shapes[img_idx]
                predictions_per_image[img_idx] = np.zeros((h, w), dtype=np.uint8)
            predictions_per_image[img_idx][x : x + args.patch_size, y : y + args.patch_size] = pred_patch

        torch.cuda.empty_cache()

# -----------------------------------------------------------------------------
# global confusion matrix
# -----------------------------------------------------------------------------
true_arr = np.concatenate(np.load(true_file, allow_pickle=True))
pred_arr = np.concatenate(np.load(pred_file, allow_pickle=True))
cm = confusion_matrix(true_arr, pred_arr)
cm_norm = confusion_matrix(true_arr, pred_arr, normalize="true")
class_names = ["Other", "Tree"]

fig_path = OUTPUT_DIR / "confusion_matrix.png"
fig_norm_path = OUTPUT_DIR / "confusion_matrix_normalized.png"

for matrix, path, title in (
    (cm, fig_path, "Confusion Matrix – Test set"),
    (cm_norm, fig_norm_path, "Normalized Confusion Matrix – Test set"),
):
    disp = ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=class_names)
    disp.plot(cmap=plt.cm.Blues)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()

print("[INFO] Confusion matrices saved.")

# -----------------------------------------------------------------------------
# save reconstructed masks
# -----------------------------------------------------------------------------
for img_idx, mask in predictions_per_image.items():
    src_path = dataset.IMG_NAMES[img_idx]
    with rasterio.open(src_path) as src:
        meta = src.meta.copy()
    meta.update({"count": 1, "dtype": "uint8", "compress": "lzw"})

    h, w = dataset.image_shapes[img_idx]
    mask = mask[:h, :w]

    image_name = Path(src_path).stem
    out_path = OUTPUT_DIR / f"{image_name}_mask.tif"
    with rasterio.open(out_path, "w", **meta) as dst:
        dst.write(mask.astype(rasterio.uint8), 1)
    print(f"[INFO] Saved mask → {out_path}")
