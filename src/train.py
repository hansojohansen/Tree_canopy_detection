#!/usr/bin/env python
"""Training script for UNet tree‑canopy segmentation.

"""

import os
import argparse
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
import torchvision.transforms as transforms
from tqdm import tqdm

from dataloader import segDataset
from losses import FocalLoss, mIoULoss
from model import UNet

# -----------------------------------------------------------------------------
# globals
# -----------------------------------------------------------------------------
threshold = 0.5  # probability threshold for class‑1

# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def build_args():
    parser = argparse.ArgumentParser(description="Train a UNet on the RGB aerial dataset.")
    parser.add_argument("--data", type=str, default="./data", help="Root directory containing 'training' & 'validating' sub‑folders")
    parser.add_argument("--num_epochs", type=int, default=2, help="Number of epochs")
    parser.add_argument("--batch", type=int, default=4, help="Batch size")
    parser.add_argument("--loss", type=str, default="crossentropy", choices=["focalloss", "iouloss", "crossentropy"], help="Loss function")
    return parser.parse_args()

# -----------------------------------------------------------------------------
# metrics helpers
# -----------------------------------------------------------------------------

def acc(label: torch.Tensor, predicted: torch.Tensor):
    """Pixel accuracy."""
    return (label.cpu() == torch.argmax(predicted, axis=1).cpu()).sum() / torch.numel(label.cpu())


def compute_metrics(label: torch.Tensor, predicted: torch.Tensor):
    """Precision / recall / F1 for binary segmentation (class 1 = tree)."""
    tp = ((label == 1) & (predicted == 1)).sum().item()
    fp = ((label == 0) & (predicted == 1)).sum().item()
    fn = ((label == 1) & (predicted == 0)).sum().item()
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    return precision, recall, f1


# -----------------------------------------------------------------------------
# main
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    args = build_args()

    transform = transforms.Compose([transforms.ToTensor()])

    train_dataset = segDataset(os.path.join(args.data, "training"), patch_size=124, mode="train", transform=transform)
    val_dataset = segDataset(os.path.join(args.data, "validating"), patch_size=124, mode="train", transform=transform)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch, shuffle=True, num_workers=0)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch, shuffle=False, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet(n_channels=3, n_classes=2, bilinear=False).to(device)

    if args.loss == "focalloss":
        criterion = FocalLoss(gamma=0.75).to(device)
    elif args.loss == "iouloss":
        criterion = mIoULoss(n_classes=2).to(device)
    else:
        criterion = nn.CrossEntropyLoss().to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = f"./saved_models/training_{timestamp}"
    os.makedirs(save_dir, exist_ok=True)

    best_val_loss = float("inf")

    # containers for plots
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    train_precisions, train_recalls, train_f1s = [], [], []
    val_precisions, val_recalls, val_f1s = [], [], []

    total_batches = len(train_loader) * args.num_epochs
    with tqdm(total=total_batches, desc="Overall progress", unit="batch") as total_pbar:
        for epoch in range(args.num_epochs):
            model.train()
            epoch_losses, epoch_accs, ep_prec, ep_rec, ep_f1 = [], [], [], [], []

            with tqdm(total=len(train_loader), desc=f"Epoch {epoch+1}/{args.num_epochs}", leave=False) as ep_pbar:
                for x, y, _, _ in train_loader:
                    preds = model(x.to(device))
                    loss = criterion(preds, y.to(device))
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    probs = torch.softmax(preds, dim=1)
                    class1_probs = probs[:, 1, :, :]
                    pred_labels = (class1_probs > threshold).long()

                    epoch_losses.append(loss.item())
                    epoch_accs.append(acc(y, preds).item())
                    pr, rc, f1 = compute_metrics(y.cpu(), pred_labels.cpu())
                    ep_prec.append(pr)
                    ep_rec.append(rc)
                    ep_f1.append(f1)

                    ep_pbar.update(1)
                    total_pbar.update(1)

            # record epoch metrics
            train_losses.append(np.mean(epoch_losses))
            train_accs.append(np.mean(epoch_accs))
            train_precisions.append(np.mean(ep_prec))
            train_recalls.append(np.mean(ep_rec))
            train_f1s.append(np.mean(ep_f1))

            # ---------------- validation ----------------
            model.eval()
            val_losses_ep, val_accs_ep, vp, vr, vf = [], [], [], [], []
            for x, y, _, _ in val_loader:
                with torch.no_grad():
                    preds = model(x.to(device))
                val_losses_ep.append(criterion(preds, y.to(device)).item())
                val_accs_ep.append(acc(y, preds).item())
                pr, rc, f1 = compute_metrics(y.cpu(), torch.argmax(preds, axis=1).cpu())
                vp.append(pr); vr.append(rc); vf.append(f1)

            val_losses.append(np.mean(val_losses_ep))
            val_accs.append(np.mean(val_accs_ep))
            val_precisions.append(np.mean(vp))
            val_recalls.append(np.mean(vr))
            val_f1s.append(np.mean(vf))

            # print summary
            print(
                f"\nEpoch {epoch+1}: train loss {train_losses[-1]:.4f} | val loss {val_losses[-1]:.4f} | "
                f"train F1 {train_f1s[-1]:.4f} | val F1 {val_f1s[-1]:.4f}"
            )

            # checkpoint
            torch.save(model.state_dict(), f"{save_dir}/model_epoch_{epoch}.pth")
            if val_losses[-1] < best_val_loss:
                best_val_loss = val_losses[-1]
                torch.save(model.state_dict(), f"{save_dir}/best_model.pth")
                print("[INFO] Best model updated.")

            scheduler.step()

    # ---------------------------------------------------------------------
    # plotting helpers
    # ---------------------------------------------------------------------

    def plot_curve(train_vals, val_vals, title, ylabel):
        plt.figure(figsize=(10, 5))
        plt.plot(train_vals, label=f"Train {ylabel}")
        plt.plot(val_vals, label=f"Val {ylabel}")
        plt.xlabel("Epoch")
        plt.ylabel(ylabel)
        plt.title(title)
        plt.legend(); plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"{save_dir}/{ylabel.lower()}_plot.png")
        plt.close()

    plot_curve(train_losses, val_losses, "Loss curve", "Loss")
    plot_curve(train_accs, val_accs, "Accuracy curve", "Accuracy")
    plot_curve(train_precisions, val_precisions, "Precision curve", "Precision")
    plot_curve(train_recalls, val_recalls, "Recall curve", "Recall")
    plot_curve(train_f1s, val_f1s, "F1 curve", "F1")

    print("Training complete. Plots and checkpoints saved to", save_dir)
