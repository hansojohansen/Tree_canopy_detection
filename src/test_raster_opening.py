#!/usr/bin/env python
"""Quick raster viewer

Opens a single GeoTIFF (first band) and displays it with a green colourmap.
"""

import argparse
from pathlib import Path

import rasterio
import matplotlib.pyplot as plt
import numpy as np

# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Display a GeoTIFF (first band) as an image.")
parser.add_argument("--tif", type=str, default="./data/testing/annotations/example.tif", help="Path to the TIFF file to display")
args = parser.parse_args()

TIFF_PATH = Path(args.tif)
print(f"Opening {TIFF_PATH}")

# -----------------------------------------------------------------------------
# read raster
# -----------------------------------------------------------------------------
with rasterio.open(TIFF_PATH) as src:
    data = src.read(1)  # first band
    nodata = src.nodata

# mask nodata
if nodata is not None:
    data = np.ma.masked_equal(data, nodata)

# -----------------------------------------------------------------------------
# plot
# -----------------------------------------------------------------------------
plt.figure(figsize=(10, 8))
plt.imshow(data, cmap="Greens", vmin=0, vmax=1)
plt.colorbar(label="Class value")
plt.title(f"{TIFF_PATH.name} – Green colour map")
plt.xlabel("Column")
plt.ylabel("Row")
plt.tight_layout()
plt.show()
