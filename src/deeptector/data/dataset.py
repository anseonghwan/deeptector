"""Frame dataset backed by video-level manifest records."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Protocol

import numpy as np
import torch
from torch.utils.data import Dataset

from .face_detection import FaceDetectionStats, FaceDetector, center_square_crop
from .manifest import VideoRecord
from .sampling import FrameSampler

LOGGER = logging.getLogger(__name__)


class VideoReader(Protocol):
    """Minimal random-access video reader contract."""

    def frame_count(self, path: str) -> int: ...

    def read(self, path: str, index: int) -> np.ndarray: ...

    def frame_rate(self, path: str) -> float: ...


class OpenCVVideoReader:
    """OpenCV-backed RGB frame reader."""

    def frame_count(self, path: str) -> int:
        import cv2

        capture = cv2.VideoCapture(path)
        try:
            if not capture.isOpened():
                raise OSError(f"Cannot open video: {path}")
            return int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        finally:
            capture.release()

    def read(self, path: str, index: int) -> np.ndarray:
        import cv2

        capture = cv2.VideoCapture(path)
        try:
            if not capture.isOpened():
                raise OSError(f"Cannot open video: {path}")
            capture.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, frame = capture.read()
            if not ok:
                raise OSError(f"Cannot read frame {index} from {path}")
            return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        finally:
            capture.release()

    def frame_rate(self, path: str) -> float:
        """Return the container-reported frame rate."""
        import cv2

        capture = cv2.VideoCapture(path)
        try:
            if not capture.isOpened():
                raise OSError(f"Cannot open video: {path}")
            fps = float(capture.get(cv2.CAP_PROP_FPS))
            if fps <= 0:
                raise OSError(f"Video reports invalid FPS {fps}: {path}")
            return fps
        finally:
            capture.release()


class VideoFrameDataset(Dataset[dict[str, object]]):
    """Flatten manifest videos into deterministically sampled face frames."""

    def __init__(
        self,
        records: list[VideoRecord],
        sampler: FrameSampler,
        transform: object,
        face_detector: FaceDetector,
        *,
        face_margin: float = 0.2,
        reader: VideoReader | None = None,
        verify_paths: bool = True,
    ) -> None:
        self.records = records
        self.transform = transform
        self.face_detector = face_detector
        self.face_margin = face_margin
        self.reader = reader or OpenCVVideoReader()
        self.face_stats = FaceDetectionStats()
        self.samples: list[tuple[int, int]] = []
        for record_index, record in enumerate(records):
            if verify_paths and not Path(record.video_path).exists():
                raise FileNotFoundError(f"Video does not exist: {record.video_path}")
            frame_count = self.reader.frame_count(record.video_path)
            indices = sampler.sample(frame_count)
            if not indices:
                LOGGER.warning("Video has no readable frames: %s", record.video_path)
            self.samples.extend((record_index, index) for index in indices)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, item: int) -> dict[str, object]:
        record_index, frame_index = self.samples[item]
        record = self.records[record_index]
        frame = self.reader.read(record.video_path, frame_index)
        self.face_stats.attempted += 1
        box = self.face_detector.detect(frame)
        face_detected = box is not None
        if box is None:
            self.face_stats.failed += 1
            crop = center_square_crop(frame)
        else:
            crop = self.face_detector.crop(frame, box, self.face_margin)
        image = self.transform(crop)
        return {
            "image": image,
            "label": torch.tensor(float(record.label), dtype=torch.float32),
            "dataset": record.dataset,
            "video_id": record.video_id,
            "video_path": record.video_path,
            "frame_index": frame_index,
            "face_detected": face_detected,
        }
