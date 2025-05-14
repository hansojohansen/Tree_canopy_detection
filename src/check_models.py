#!/usr/bin/env python
"""Utility: inspect a `.pth` checkpoint file.

Prints the Python type of the loaded object and its top-level keys. Useful for
quick sanity‐checks of saved Torch models.
"""

import argparse
from pathlib import Path
import torch

# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Inspect a PyTorch .pth checkpoint file.")
parser.add_argument("--model_path", type=str, default="./saved_models/best_model.pth", help="Path to .pth file")
args = parser.parse_args()

ckpt_path = Path(args.model_path)
print(f"Loading {ckpt_path}")

obj = torch.load(ckpt_path, map_location="cpu")
print("Loaded object type:", type(obj))

if isinstance(obj, dict):
    print("Top-level keys and their types:")
    for k, v in obj.items():
        print(f"  {k}: {type(v)}")
else:
    print(obj)
