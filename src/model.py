#!/usr/bin/env python
"""UNet implementation (original bachelor-thesis variant).

Architecture
------------
Encoder feature depths: 64 → 128 → 256 → 512 → 1024  
Decoder mirrors encoder; optional bilinear up-sampling.

"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

# -----------------------------------------------------------------------------  
# Building blocks
# -----------------------------------------------------------------------------


class DoubleConv(nn.Module):
    """Two consecutive `Conv2d → BatchNorm → ReLU` layers."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        mid_channels: int | None = None,
    ) -> None:
        super().__init__()
        mid = out_channels if mid_channels is None else mid_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
        return self.double_conv(x)


class Down(nn.Module):
    """Down-scaling (max-pool) followed by `DoubleConv`."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2), DoubleConv(in_channels, out_channels)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Up-scaling then `DoubleConv`.

    Parameters
    ----------
    in_channels : int
        Number of input feature channels.
    out_channels : int
        Number of output feature channels.
    bilinear : bool, default ``False``
        If *True*, use `nn.Upsample`; otherwise, `ConvTranspose2d`.
    """

    def __init__(self, in_channels: int, out_channels: int, bilinear: bool = False) -> None:
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:  # transpose-conv
            self.up = nn.ConvTranspose2d(
                in_channels, in_channels // 2, kernel_size=2, stride=2
            )
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:  # noqa: D401
        x1 = self.up(x1)
        # Pad to handle odd input dimensions
        diff_y = x2.size(2) - x1.size(2)
        diff_x = x2.size(3) - x1.size(3)
        x1 = F.pad(
            x1,
            [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2],
        )
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    """Final `1 × 1` convolution producing class-logits."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
        return self.conv(x)


# -----------------------------------------------------------------------------  
# UNet main
# -----------------------------------------------------------------------------


class UNet(nn.Module):
    """Vanilla UNet (bachelor-thesis variant)."""

    def __init__(
        self,
        n_channels: int,
        n_classes: int,
        bilinear: bool = True,
    ) -> None:
        super().__init__()
        factor = 2 if bilinear else 1  # halve feature-depth in decoder if bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024 // factor)

        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    # ----------------------------- forward -----------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        return self.outc(x)

    # ------------------------ optional memory tricks -------------------------

    def use_checkpointing(self) -> None:
        """Wrap layers with `torch.utils.checkpoint` to trade compute for memory."""
        import torch.utils.checkpoint as cp

        self.inc = cp.checkpoint(self.inc)
        self.down1 = cp.checkpoint(self.down1)
        self.down2 = cp.checkpoint(self.down2)
        self.down3 = cp.checkpoint(self.down3)
        self.down4 = cp.checkpoint(self.down4)
        self.up1 = cp.checkpoint(self.up1)
        self.up2 = cp.checkpoint(self.up2)
        self.up3 = cp.checkpoint(self.up3)
        self.up4 = cp.checkpoint(self.up4)
        self.outc = cp.checkpoint(self.outc)
