#!/usr/bin/env python
"""Dataset loader for aerial‐imagery segmentation.

* Splits each image (and optional annotation) into fixed‑size non‑overlapping
  patches.
* Supports `train` mode (expects matching `annotations/` folder) and
  `inference` mode (images only).

"""

from __future__ import annotations
import os
from glob import glob
import random  # retained for potential future augmentations

import numpy as np
import torch
from rasterio.crs import CRS
import rasterio
import matplotlib.pyplot as plt  # noqa: F401 (import kept in case of debug plotting)


class segDataset(torch.utils.data.Dataset):
    """Patch‑based dataset for segmentation models."""

    def __init__(
        self,
        root: str,
        patch_size: int = 124,
        mode: str = "inference",
        crs: CRS | None = None,
        res: float | None = None,
        transform=None,
    ) -> None:
        super().__init__()
        self.root = root
        self.patch_size = patch_size
        self.mode = mode  # "train" (images+annotations) or "inference" (images only)
        self.transform = transform
        self.crs = CRS.from_epsg(25833) if crs is None else crs  # default UTM 33N
        self.res = 0.25 if res is None else res  # default 25 cm resolution

        images_path = os.path.join(self.root, "images", "*.tif")
        self.IMG_NAMES = sorted(glob(images_path))
        print(f"Loaded {len(self.IMG_NAMES)} images from {images_path}")

        # containers
        self.image_patches: list[np.ndarray] = []
        self.annotation_patches: list[np.ndarray] | None = [] if self.mode == "train" else None
        self.positions: list[tuple[int, int]] = []
        self.image_indices: list[int] = []
        self.image_shapes: list[tuple[int, int]] = []

        self._create_patches()

    # ---------------------------------------------------------------------
    # internal helpers
    # ---------------------------------------------------------------------

    def _create_patches(self):
        for img_idx, img_path in enumerate(self.IMG_NAMES):
            # locate matching annotation file (if in train mode)
            if self.mode == "train":
                anno_path = img_path.replace("images", "annotations")
                if not os.path.exists(anno_path):
                    print(f"[WARN] Annotation not found for {img_path} – skipped.")
                    continue

            # read image
            with rasterio.open(img_path) as src_img:
                image = src_img.read().transpose(1, 2, 0)  # to H,W,C
                self.image_shapes.append(image.shape[:2])

            # read annotation
            annotation = None
            if self.mode == "train":
                with rasterio.open(anno_path) as src_anno:
                    annotation = src_anno.read(1)

            # generate patches
            img_p, anno_p, pos = self.crop_to_patches(image, annotation)
            self.image_patches.extend(img_p)
            if self.mode == "train":
                self.annotation_patches.extend(anno_p)  # type: ignore[arg-type]
            self.positions.extend(pos)
            self.image_indices.extend([img_idx] * len(img_p))

        print(f"Generated {len(self.image_patches)} patches from dataset {self.root}")

    def crop_to_patches(self, image: np.ndarray, annotation: np.ndarray | None = None):
        patches, anno_patches, positions = [], [], []
        h, w, _ = image.shape
        stride = self.patch_size
        for i in range(0, h - stride + 1, stride):
            for j in range(0, w - stride + 1, stride):
                img_patch = image[i : i + stride, j : j + stride]
                if img_patch.shape[:2] != (stride, stride):
                    print(f"[INFO] skipping patch at ({i},{j}) – size mismatch")
                    continue
                patches.append(img_patch)
                positions.append((i, j))
                if annotation is not None:
                    anno_patch = annotation[i : i + stride, j : j + stride]
                    anno_patches.append(anno_patch)
        return patches, anno_patches if annotation is not None else None, positions

    # ------------------------------------------------------------------
    # PyTorch Dataset interface
    # ------------------------------------------------------------------

    def __getitem__(self, idx):
        img_patch = self.image_patches[idx]          # H × W × C, uint8
        pos       = self.positions[idx]
        img_idx   = self.image_indices[idx]

        # --- convert to Torch tensor -------------------------------------------------
        # 1.  float32  [& optional 0-1 scaling]
        img_patch = img_patch.astype(np.float32) / 255.0
        # 2.  channel-first  (C, H, W)
        img_patch = np.moveaxis(img_patch, -1, 0)
        # 3.  torch tensor
        img_patch = torch.from_numpy(img_patch)      # dtype = float32

        if self.mode == "train":
            anno_patch = self.annotation_patches[idx].astype(np.int64)
            return (
                img_patch,                           # float32  (C, H, W)
                torch.from_numpy(anno_patch),        # int64    (H, W)
                pos,
                img_idx,
            )
        else:
            return img_patch, pos, img_idx


    def __len__(self):
        return len(self.image_patches)
