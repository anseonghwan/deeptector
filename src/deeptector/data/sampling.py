"""Deterministic frame-index sampling."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class FrameSampler(ABC):
    """Replaceable frame sampler interface."""

    @abstractmethod
    def sample(self, frame_count: int) -> list[int]:
        """Return ordered zero-based frame indices."""


class UniformFrameSampler(FrameSampler):
    """Evenly space frames, deterministically, over the complete video."""

    def __init__(self, frames_per_video: int) -> None:
        if frames_per_video < 1:
            raise ValueError("frames_per_video must be positive")
        self.frames_per_video = frames_per_video

    def sample(self, frame_count: int) -> list[int]:
        if frame_count < 1:
            return []
        count = min(frame_count, self.frames_per_video)
        return np.linspace(0, frame_count - 1, num=count, dtype=int).tolist()
