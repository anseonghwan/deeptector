"""Replaceable, deterministic face detection and cropping."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

Box = tuple[int, int, int, int]


class FaceDetector(ABC):
    """Face location and crop interface."""

    @abstractmethod
    def detect(self, frame: np.ndarray) -> Box | None:
        """Return (x1, y1, x2, y2), or None when no face is found."""

    def crop(self, frame: np.ndarray, box: Box, margin: float = 0.2) -> np.ndarray:
        """Crop a square region around a face with bounded proportional margin."""
        if margin < 0:
            raise ValueError("margin cannot be negative")
        height, width = frame.shape[:2]
        x1, y1, x2, y2 = box
        side = max(x2 - x1, y2 - y1) * (1 + 2 * margin)
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        left = max(0, round(cx - side / 2))
        top = max(0, round(cy - side / 2))
        right = min(width, round(cx + side / 2))
        bottom = min(height, round(cy + side / 2))
        if right <= left or bottom <= top:
            raise ValueError(f"Invalid face box after clipping: {box}")
        return frame[top:bottom, left:right]


class OpenCVHaarFaceDetector(FaceDetector):
    """Lightweight OpenCV detector; the largest detected face is selected."""

    def __init__(self) -> None:
        import cv2

        path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self._classifier = cv2.CascadeClassifier(path)
        if self._classifier.empty():
            raise RuntimeError(f"Could not load OpenCV face cascade: {path}")

    def detect(self, frame: np.ndarray) -> Box | None:
        import cv2

        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        faces = self._classifier.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
        if len(faces) == 0:
            return None
        x, y, width, height = max(faces, key=lambda item: int(item[2]) * int(item[3]))
        return int(x), int(y), int(x + width), int(y + height)


@dataclass
class FaceDetectionStats:
    """Observable face preprocessing counters."""

    attempted: int = 0
    failed: int = 0

    @property
    def failure_rate(self) -> float:
        return self.failed / self.attempted if self.attempted else 0.0


def center_square_crop(frame: np.ndarray) -> np.ndarray:
    """Deterministic fallback that preserves content aspect ratio before resize."""
    height, width = frame.shape[:2]
    side = min(height, width)
    top, left = (height - side) // 2, (width - side) // 2
    return frame[top : top + side, left : left + side]
