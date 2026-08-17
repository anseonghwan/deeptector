"""Embedding-preserving binary detector."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .base import VisualEncoder


@dataclass
class DetectorOutput:
    """Detector result. Score is an uncalibrated sigmoid score."""

    embedding: torch.Tensor
    logits: torch.Tensor
    score: torch.Tensor


class DeepfakeClassifier(nn.Module):
    """Visual encoder plus a separate LayerNorm classification head."""

    def __init__(self, encoder: VisualEncoder, hidden_dim: int | None = None) -> None:
        super().__init__()
        self.encoder = encoder
        if hidden_dim is None:
            self.head = nn.Sequential(
                nn.LayerNorm(encoder.output_dim), nn.Linear(encoder.output_dim, 1)
            )
        else:
            self.head = nn.Sequential(
                nn.LayerNorm(encoder.output_dim),
                nn.Linear(encoder.output_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, 1),
            )

    def forward(self, images: torch.Tensor) -> DetectorOutput:
        embedding = self.encoder(images)
        if embedding.ndim != 2:
            raise ValueError(f"Encoder must return [B, D], got {tuple(embedding.shape)}")
        logits = self.head(embedding).squeeze(-1)
        return DetectorOutput(embedding=embedding, logits=logits, score=torch.sigmoid(logits))
