"""Visual encoder contracts."""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from torch import nn


class VisualEncoder(nn.Module, ABC):
    """An image encoder returning one [B, D] embedding per image."""

    output_dim: int

    @abstractmethod
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Encode BCHW image tensors."""
