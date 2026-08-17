import numpy as np
import torch
from torch import nn

from deeptector.data.face_detection import FaceDetector
from deeptector.models.base import VisualEncoder


class ArrayReader:
    def __init__(self, videos):
        self.videos = videos

    def frame_count(self, path):
        return len(self.videos[path])

    def read(self, path, index):
        return self.videos[path][index]


class FullFrameDetector(FaceDetector):
    def detect(self, frame: np.ndarray):
        height, width = frame.shape[:2]
        return 0, 0, width, height


class TinyEncoder(VisualEncoder):
    output_dim = 6

    def __init__(self, frozen=False):
        super().__init__()
        self.projection = nn.Linear(3, self.output_dim)
        for parameter in self.parameters():
            parameter.requires_grad = not frozen

    def forward(self, images: torch.Tensor):
        return self.projection(images.mean(dim=(2, 3)))
