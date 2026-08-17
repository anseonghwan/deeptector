"""Deterministic image transforms."""

from __future__ import annotations

import numpy as np
import torch
from PIL import Image

CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


class ImageTransform:
    """Resize RGB arrays and normalize them into CHW tensors."""

    def __init__(self, resolution: int = 224, normalization: str = "clip") -> None:
        self.resolution = resolution
        if normalization not in {"clip", "imagenet", "none"}:
            raise ValueError(f"Unknown normalization: {normalization}")
        self.normalization = normalization

    def __call__(self, image: np.ndarray) -> torch.Tensor:
        resized = Image.fromarray(image.astype(np.uint8), mode="RGB").resize(
            (self.resolution, self.resolution), Image.Resampling.BICUBIC
        )
        tensor = torch.from_numpy(np.asarray(resized).copy()).permute(2, 0, 1).float() / 255.0
        if self.normalization != "none":
            mean, std = (
                (CLIP_MEAN, CLIP_STD)
                if self.normalization == "clip"
                else ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            )
            tensor = (tensor - torch.tensor(mean)[:, None, None]) / torch.tensor(std)[:, None, None]
        return tensor
