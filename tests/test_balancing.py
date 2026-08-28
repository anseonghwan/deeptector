import pytest
import torch
from torch.utils.data import RandomSampler, SequentialSampler, WeightedRandomSampler

from deeptector.cli import common
from deeptector.data.manifest import VideoRecord


class ManifestOnlyDataset(torch.utils.data.Dataset):
    def __init__(self, records, *_args, **_kwargs):
        self.records = records
        self.samples = [(index, 0) for index in range(len(records))]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        record_index, _ = self.samples[index]
        return {"label": torch.tensor(float(self.records[record_index].label))}


def _records(split, real=25, fake=75):
    return [
        VideoRecord(f"{split}-real-{index}", "unused", "fixture", split, 0) for index in range(real)
    ] + [
        VideoRecord(f"{split}-fake-{index}", "unused", "fixture", split, 1) for index in range(fake)
    ]


def _patch_dataset(monkeypatch, records):
    monkeypatch.setattr(common, "load_manifest", lambda _path: records)
    monkeypatch.setattr(common, "VideoFrameDataset", ManifestOnlyDataset)
    monkeypatch.setattr(common, "OpenCVHaarFaceDetector", lambda: object())


def _loader(monkeypatch, split, records, **kwargs):
    _patch_dataset(monkeypatch, records)
    return common.build_loader(
        "fixture.csv",
        {"frames_per_video": 1, "input_resolution": 8},
        split=split,
        batch_size=10,
        **kwargs,
    )


def test_train_balancing_weights_and_effective_sampling_are_label_derived(monkeypatch):
    records = _records("train")
    loader = _loader(monkeypatch, "train", records, balanced_sampling=True, seed=42)
    assert isinstance(loader.sampler, WeightedRandomSampler)
    weights = loader.sampler.weights.tolist()
    assert weights[:25] == pytest.approx([1 / 25] * 25)
    assert weights[25:] == pytest.approx([1 / 75] * 75)
    sampled_labels = [records[index].label for index in list(loader.sampler)]
    assert abs(sampled_labels.count(0) - sampled_labels.count(1)) <= 20


def test_balanced_sampler_is_deterministic_for_same_seed(monkeypatch):
    records = _records("train")
    first = _loader(monkeypatch, "train", records, balanced_sampling=True, seed=7)
    first_indices = list(first.sampler)
    second = _loader(monkeypatch, "train", records, balanced_sampling=True, seed=7)
    assert first_indices == list(second.sampler)


@pytest.mark.parametrize("split", ["validation", "test"])
def test_validation_and_test_preserve_official_order(monkeypatch, split):
    records = _records(split)
    loader = _loader(monkeypatch, split, records, balanced_sampling=False)
    assert isinstance(loader.sampler, SequentialSampler)
    labels = [int(batch["label"].item()) for batch in loader.dataset]
    assert labels.count(0) == 25
    assert labels.count(1) == 75


def test_balancing_is_rejected_outside_train(monkeypatch):
    records = _records("validation")
    with pytest.raises(ValueError, match="only for the train split"):
        _loader(monkeypatch, "validation", records, balanced_sampling=True)


def test_disabling_balancing_preserves_original_shuffle_behavior(monkeypatch):
    records = _records("train")
    loader = _loader(monkeypatch, "train", records, balanced_sampling=False, shuffle=True)
    assert isinstance(loader.sampler, RandomSampler)
    assert not isinstance(loader.sampler, WeightedRandomSampler)
