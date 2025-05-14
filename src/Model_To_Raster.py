#!/usr/bin/env python
"""Model‑to‑Raster Inference Script

Runs inference with a trained U‑Net on the *testing* subset and writes:
  • georeferenced mask TIFFs
  • per‑image confusion matrices (PNG + txt metrics)

"""

import os
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import rasterio
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay  # kept for parity even if unused directly
import matplotlib.pyplot as plt
import seaborn as sns

# local modules
from dataloader import segDataset
from model import UNet

# -----------------------------------------------------------------------------
# default constants
# -----------------------------------------------------------------------------
B_VALUE = True        # bilinear upsampling flag in UNet
TILE_SIZE = 124
DEFAULT_BATCH = 6

# -----------------------------------------------------------------------------
# device
# -----------------------------------------------------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Run inference and export raster masks + metrics.")
parser.add_argument("--data_root", type=str, default="./data", help="Root folder containing a 'testing' sub‑directory with images and annotations.")
parser.add_argument("--model_path", type=str, default="./saved_models/best_model.pth", help="Trained model checkpoint (\.pth)")
parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH, help="Batch size for DataLoader")
parser.add_argument("--patch_size", type=int, default=TILE_SIZE, help="Patch (tile) size in pixels")
parser.add_argument("--threshold", type=float, default=0.5, help="Probability threshold for class‑1 (tree canopy)")
args = parser.parse_args()

# -----------------------------------------------------------------------------
# paths & dataset
# -----------------------------------------------------------------------------
TESTING_ROOT = os.path.join(args.data_root, "testing")
print(f"Testing data root: {TESTING_ROOT}")

dataset = segDataset(root=TESTING_ROOT, patch_size=args.patch_size, mode="train", transform=transforms.Compose([]))

def custom_collate_fn(batch):
    """Keeps (images, annotations, positions, image_indices) together."""
    if not batch:
        return None
    images, annotations, positions, image_indices = zip(*batch)
    return torch.stack(images, 0), torch.stack(annotations, 0), list(positions), list(image_indices)

loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=custom_collate_fn)

# -----------------------------------------------------------------------------
# model
# -----------------------------------------------------------------------------
model = UNet(n_channels=3, n_classes=2, bilinear=B_VALUE).to(DEVICE)
model.load_state_dict(torch.load(args.model_path, map_location=DEVICE))
model.eval()

# -----------------------------------------------------------------------------
# output dirs
# -----------------------------------------------------------------------------
model_name = os.path.basename(os.path.dirname(args.model_path)) or "model"
OUTPUT_ROOT = os.path.join("outputs", "predicted_masks", model_name)
os.makedirs(OUTPUT_ROOT, exist_ok=True)

# -----------------------------------------------------------------------------
# containers
# -----------------------------------------------------------------------------
cm_per_image = {}
predictions_per_image = {}

# -----------------------------------------------------------------------------
# inference
# -----------------------------------------------------------------------------
with torch.no_grad():
    for batch in loader:
        if batch is None:
            continue
        inputs, annotations, positions, image_indices = batch
        inputs = inputs.to(DEVICE)
        outputs = model(inputs)
        probs = torch.softmax(outputs, dim=1)[:, 1, :, :]  # class‑1 probs
        preds = (probs > args.threshold).long().cpu().numpy()
        annotations_np = annotations.cpu().numpy()

        for i in range(inputs.size(0)):
            pred_patch = preds[i]
            true_patch = annotations_np[i]
            x, y = positions[i]
            img_idx = image_indices[i]
            img_path = dataset.IMG_NAMES[img_idx]
            image_name = os.path.splitext(os.path.basename(img_path))[0]

            # accumulate confusion matrix
            cm_batch = confusion_matrix(true_patch.flatten(), pred_patch.flatten(), labels=[0, 1])
            cm_per_image.setdefault(image_name, np.zeros((2, 2), dtype=np.int64))
            cm_per_image[image_name] += cm_batch

            # stitch patches back into full mask
            if img_idx not in predictions_per_image:
                h, w = dataset.image_shapes[img_idx]
                predictions_per_image[img_idx] = np.zeros((h, w), dtype=np.uint8)
            predictions_per_image[img_idx][x : x + args.patch_size, y : y + args.patch_size] = pred_patch

# -----------------------------------------------------------------------------
# save confusion matrices + metrics
# -----------------------------------------------------------------------------
for image_name, cm_total in cm_per_image.items():
    tp = cm_total[1, 1]
    tn = cm_total[0, 0]
    fp = cm_total[0, 1]
    fn = cm_total[1, 0]

    accuracy = (tp + tn) / cm_total.sum()
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)

    image_dir = os.path.join(OUTPUT_ROOT, image_name)
    os.makedirs(image_dir, exist_ok=True)

    # Confusion matrix plot
    cm_norm = cm_total.astype(float) / cm_total.sum(axis=1, keepdims=True)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_norm, annot=True, fmt=".2%", cmap="Blues", xticklabels=["Predicted Other", "Predicted Tree"], yticklabels=["Actual Other", "Actual Tree"])
    plt.title(f"Confusion Matrix – {image_name}")
    plt.ylabel("Ground truth")
    plt.xlabel("Model prediction")
    plt.tight_layout()
    plt.savefig(os.path.join(image_dir, "confusion_matrix.png"))
    plt.close()

    # metrics text file
    with open(os.path.join(image_dir, "metrics.txt"), "w") as f:
        f.write(f"Accuracy : {accuracy*100:.2f}%\n")
        f.write(f"Precision: {precision*100:.2f}%\n")
        f.write(f"Recall   : {recall*100:.2f}%\n")
        f.write(f"F1 Score : {f1*100:.2f}%\n")
        f.write("\nConfusion Matrix (counts):\n")
        f.write(np.array2string(cm_total, separator=", "))

    print(f"[INFO] Metrics saved for {image_name}")

# -----------------------------------------------------------------------------
# save masks as georeferenced TIFFs
# -----------------------------------------------------------------------------
for img_idx, mask in predictions_per_image.items():
    src_path = dataset.IMG_NAMES[img_idx]
    with rasterio.open(src_path) as src:
        meta = src.meta.copy()
    meta.update({"count": 1, "dtype": "uint8", "compress": "lzw"})

    h, w = dataset.image_shapes[img_idx]
    mask = mask[:h, :w]

    image_name = os.path.splitext(os.path.basename(src_path))[0]
    out_dir = os.path.join(OUTPUT_ROOT, image_name)
    os.makedirs(out_dir, exist_ok=True)
    out_tif = os.path.join(out_dir, f"{image_name}_mask.tif")

    with rasterio.open(out_tif, "w", **meta) as dst:
        dst.write(mask.astype(rasterio.uint8), 1)

    print(f"[INFO] Saved mask → {out_tif}")
