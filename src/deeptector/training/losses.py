"""Milestone 1 losses."""

from torch import nn


def binary_classification_loss() -> nn.Module:
    """Return the transparent logit-space binary objective."""
    return nn.BCEWithLogitsLoss()
