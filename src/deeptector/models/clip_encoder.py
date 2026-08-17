"""Hugging Face CLIP vision encoder adapter."""

from __future__ import annotations

import torch

from .base import VisualEncoder


class CLIPVisualEncoder(VisualEncoder):
    """Pretrained CLIP visual tower with optional frozen parameters."""

    def __init__(
        self,
        model_name: str = "openai/clip-vit-base-patch16",
        *,
        freeze: bool = True,
        local_files_only: bool = False,
    ) -> None:
        super().__init__()
        try:
            from transformers import CLIPVisionModel
        except ImportError as error:
            raise ImportError("Install deeptector dependencies to use CLIPVisualEncoder") from error
        self.backbone = CLIPVisionModel.from_pretrained(
            model_name, local_files_only=local_files_only
        )
        self.output_dim = int(self.backbone.config.hidden_size)
        self.freeze(freeze)

    def freeze(self, frozen: bool = True) -> None:
        """Set the full visual tower's trainability."""
        for parameter in self.backbone.parameters():
            parameter.requires_grad = not frozen
        if frozen:
            self.backbone.eval()

    def train(self, mode: bool = True) -> CLIPVisualEncoder:
        super().train(mode)
        if not any(parameter.requires_grad for parameter in self.backbone.parameters()):
            self.backbone.eval()
        return self

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.backbone(pixel_values=images).pooler_output
