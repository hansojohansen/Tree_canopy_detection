#!/usr/bin/env python
"""Overlay predictions on original imagery

Creates RGB overlays of predicted tree‑cover masks for quick visual inspection.
"""

import os
import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import matplotlib.pyplot as plt  # noqa: F401  (import retained: user may plot later)
import cv2
import rasterio

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
    description="Run inference (train/inference mode) and save RGB overlays of predicted masks.")
parser.add_argument("--data_root", type=str, default="./data", help="Root folder containing a 'testing' subdir")
parser.add_argument("--model_path", type=str, default="./saved_models/best_model.pth", help="Path to trained model .pth")
parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
parser.add_argument("--patch_size", type=int, default=512, help="Tile size in pixels")
parser.add_argument("--threshold", type=float, default=0.5, help="Class‑1 probability threshold")
parser.add_argument("--mode", type=str, default="inference", choices=["train", "inference"], help="Dataset mode")
parser.add_argument("--out_dir", type=str, default="outputs/overlayed_images", help="Where to save overlays")
args = parser.parse_args()

# -----------------------------------------------------------------------------
# dataset & loader
# -----------------------------------------------------------------------------
TEST_ROOT = Path(args.data_root) / "testing"
print(f"Testing data root: {TEST_ROOT}")

dataset = segDataset(root=str(TEST_ROOT), patch_size=args.patch_size, mode=args.mode, transform=transforms.Compose([]))


def collate(batch):
    if not batch:
        return None
    # batch item = (img, anno?, pos, img_idx)
    if args.mode == "train":
        images, annotations, positions, idxs = zip(*batch)
        return torch.stack(images), torch.stack(annotations), list(positions), list(idxs)
    images, positions, idxs = zip(*batch)
    return torch.stack(images), list(positions), list(idxs)

loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=collate)

# -----------------------------------------------------------------------------
# model
# -----------------------------------------------------------------------------
model = UNet(n_channels=3, n_classes=2, bilinear=True).to(DEVICE)
model.load_state_dict(torch.load(args.model_path, map_location=DEVICE))
model.eval()

# -----------------------------------------------------------------------------
# inference & patch stitching
# -----------------------------------------------------------------------------
predictions_per_image = {}

with torch.no_grad():
    for batch in loader:
        if batch is None:
            continue
        if args.mode == "train":
            imgs, _, pos, idxs = batch
        else:
            imgs, pos, idxs = batch
        imgs = imgs.to(DEVICE)
        probs = torch.softmax(model(imgs), dim=1)[:, 1, :, :]
        preds = (probs > args.threshold).long().cpu().numpy()

        for i in range(imgs.size(0)):
            x, y = pos[i]
            img_idx = idxs[i]
            if img_idx not in predictions_per_image:
                h, w = dataset.image_shapes[img_idx]
                predictions_per_image[img_idx] = np.zeros((h, w), dtype=np.uint8)
            predictions_per_image[img_idx][x : x + args.patch_size, y : y + args.patch_size] = preds[i]

# -----------------------------------------------------------------------------
# overlay & save
# -----------------------------------------------------------------------------
OUT_DIR = Path(args.out_dir)
OUT_DIR.mkdir(parents=True, exist_ok=True)

# -------------------------------------------------------------------
# build green overlays and write PNGs
# -------------------------------------------------------------------
for img_idx, mask in predictions_per_image.items():
    img_path = dataset.IMG_NAMES[img_idx]

    # --- NEW: read the *.tif with rasterio instead of cv2 -----------------
    with rasterio.open(img_path) as src:
        # stack bands → H × W × 3  (assumes RGB TIFF)
        img_arr = np.stack([src.read(i + 1) for i in range(src.count)], axis=-1)
    # scale to uint8 for display / OpenCV
    image = (img_arr / img_arr.max() * 255).astype(np.uint8)
    # ----------------------------------------------------------------------

    # make sure the image matches the mask size
    h, w = dataset.image_shapes[img_idx]
    if (image.shape[0], image.shape[1]) != (h, w):
        image = cv2.resize(image, (w, h))

    # green-tinted mask overlay
    color_mask = np.zeros_like(image)
    color_mask[mask == 1] = [0, 255, 0]

    alpha = 0.5
    blended = cv2.addWeighted(image, 1 - alpha, color_mask, alpha, 0)

    overlay = image.copy()
    overlay[mask == 1] = blended[mask == 1]

    out_path = OUT_DIR / f"{Path(img_path).stem}_overlay.png"
    cv2.imwrite(str(out_path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    print(f"[INFO] Overlay saved → {out_path}")
