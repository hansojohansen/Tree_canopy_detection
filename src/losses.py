#!/usr/bin/env python
"""Custom loss functions for semantic segmentation

* **FocalLoss** – focuses training on hard pixels; useful for class‑imbalance.
* **mIoULoss**  – returns `1 − mean(IoU)` across classes, a differentiable
  approximation of the common evaluation metric.

"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

# -----------------------------------------------------------------------------
# Focal Loss
# -----------------------------------------------------------------------------

class FocalLoss(nn.Module):
    """Focal loss for dense classification / segmentation.

    Args:
        gamma (float): focusing parameter; higher => more focus on hard pixels.
        alpha (float | list | None): class weighting. If *float*, interpreted as
            weight for class‑0; class‑1 weight becomes `1‑alpha`.
        size_average (bool): return mean (True) or sum (False).
    """

    def __init__(self, gamma: float = 0.0, alpha=None, size_average: bool = True):
        super().__init__()
        self.gamma = gamma
        self.size_average = size_average
        if isinstance(alpha, (float, int)):
            self.alpha = torch.tensor([alpha, 1 - alpha])
        elif isinstance(alpha, list):
            self.alpha = torch.tensor(alpha)
        else:
            self.alpha = None

    def forward(self, input: torch.Tensor, target: torch.Tensor):  # noqa: D401
        """Compute focal loss (pixel‑wise)."""
        if input.dim() > 2:
            # N,C,H,W -> N*H*W,C
            input = input.view(input.size(0), input.size(1), -1)
            input = input.transpose(1, 2).contiguous().view(-1, input.size(1))
        target = target.view(-1, 1)

        logpt = F.log_softmax(input, dim=-1).gather(1, target).view(-1)
        pt = logpt.exp()

        if self.alpha is not None:
            self.alpha = self.alpha.to(input.device)
            at = self.alpha.gather(0, target.view(-1))
            logpt = logpt * at

        loss = -((1 - pt) ** self.gamma) * logpt
        return loss.mean() if self.size_average else loss.sum()

# -----------------------------------------------------------------------------
# mean‑IoU Loss (1 − IoU)
# -----------------------------------------------------------------------------

class mIoULoss(nn.Module):
    """`1 − mean(IoU)` loss over *n_classes* for segmentation tasks."""

    def __init__(self, n_classes: int = 2):
        super().__init__()
        self.n_classes = n_classes

    @staticmethod
    def to_one_hot(tensor: torch.Tensor, num_classes: int) -> torch.Tensor:
        n, h, w = tensor.size()
        return torch.zeros(n, num_classes, h, w, device=tensor.device).scatter_(1, tensor.view(n, 1, h, w), 1)

    def forward(self, inputs: torch.Tensor, target: torch.Tensor):  # noqa: D401
        """Compute loss value."""
        n = inputs.size(0)
        probs = F.softmax(inputs, dim=1)
        target_1h = self.to_one_hot(target, self.n_classes)

        inter = (probs * target_1h).view(n, self.n_classes, -1).sum(2)
        union = (probs + target_1h - probs * target_1h).view(n, self.n_classes, -1).sum(2)
        iou = inter / union
        return 1 - iou.mean()
