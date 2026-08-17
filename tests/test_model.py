import torch
from helpers import TinyEncoder

from deeptector.models.classifier import DeepfakeClassifier


def test_model_exposes_embedding_logit_and_score():
    model = DeepfakeClassifier(TinyEncoder())
    output = model(torch.randn(4, 3, 8, 8))
    assert output.embedding.shape == (4, 6)
    assert output.logits.shape == (4,)
    assert output.score.shape == (4,)


def test_frozen_backbone_parameters_stay_frozen():
    encoder = TinyEncoder(frozen=True)
    model = DeepfakeClassifier(encoder)
    assert not any(parameter.requires_grad for parameter in model.encoder.parameters())
    assert all(parameter.requires_grad for parameter in model.head.parameters())


def test_cpu_inference():
    model = DeepfakeClassifier(TinyEncoder()).cpu().eval()
    with torch.no_grad():
        output = model(torch.zeros(2, 3, 8, 8))
    assert output.embedding.device.type == "cpu"
    assert torch.isfinite(output.score).all()
