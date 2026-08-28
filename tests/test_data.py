import numpy as np
from helpers import ArrayReader, FullFrameDetector

from deeptector.data.dataset import VideoFrameDataset
from deeptector.data.face_detection import OpenCVHaarFaceDetector
from deeptector.data.manifest import VideoRecord
from deeptector.data.sampling import UniformFrameSampler
from deeptector.data.transforms import ImageTransform


def test_uniform_sampling_is_deterministic():
    sampler = UniformFrameSampler(4)
    assert sampler.sample(10) == [0, 3, 6, 9]
    assert sampler.sample(10) == sampler.sample(10)


def test_dataset_output_shape_without_video_files():
    frame = np.full((10, 16, 3), 127, dtype=np.uint8)
    reader = ArrayReader({"memory": [frame, frame]})
    dataset = VideoFrameDataset(
        [VideoRecord("v", "memory", "fixture", "train", 1)],
        UniformFrameSampler(2),
        ImageTransform(8),
        FullFrameDetector(),
        reader=reader,
        verify_paths=False,
    )
    sample = dataset[0]
    assert sample["image"].shape == (3, 8, 8)
    assert sample["label"].item() == 1.0
    assert sample["frame_index"] == 0


def test_opencv_haar_detector_is_available():
    detector = OpenCVHaarFaceDetector()
    assert detector is not None
