#!/usr/bin/env python
"""Compute class balance in annotation rasters.

Scans a directory of annotation TIFFs, counts pixels per class and prints the
percentage breakdown for each file and overall.
"""

from __future__ import annotations
import os
import glob
from pathlib import Path
import numpy as np
import rasterio

# -----------------------------------------------------------------------------
# main helper
# -----------------------------------------------------------------------------

def calculate_class_percentages(annotation_dir: str | Path, class_labels: dict[int, str] | None = None) -> None:
    """Print per-file and global class-percentage statistics."""
    if class_labels is None:
        class_labels = {0: "Other", 1: "Tree"}

    annotation_dir = Path(annotation_dir)
    files = list(annotation_dir.glob("*.tif"))
    print(f"Found {len(files)} annotation TIFFs in {annotation_dir}")

    overall_counts = {cls: 0 for cls in class_labels}
    overall_total = 0

    for tif in files:
        with rasterio.open(tif) as src:
            data = src.read(1).flatten()
        unique, counts = np.unique(data, return_counts=True)
        per_cls = dict(zip(unique, counts))

        total_px = counts.sum()
        overall_total += total_px
        for cls in class_labels:
            overall_counts[cls] += per_cls.get(cls, 0)

        # per-file printout
        print(f"\nFile: {tif.name}")
        print(f"  Total pixels: {total_px}")
        for cls, name in class_labels.items():
            pct = per_cls.get(cls, 0) / total_px * 100 if total_px else 0
            print(f"  Class '{name}' ({cls}): {per_cls.get(cls, 0)} pixels ({pct:.2f}%)")

    # overall stats
    print("\nGlobal class proportions:")
    print(f"  Total pixels: {overall_total}")
    for cls, name in class_labels.items():
        pct = overall_counts[cls] / overall_total * 100 if overall_total else 0
        print(f"  Class '{name}' ({cls}): {overall_counts[cls]} pixels ({pct:.2f}%)")

# -----------------------------------------------------------------------------
# CLI wrapper
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Compute class-balance statistics in annotation rasters.")
    parser.add_argument("--anno_dir", type=str, default="./data/training/annotations", help="Directory with annotation TIFFs")
    args = parser.parse_args()

    calculate_class_percentages(args.anno_dir)
