from __future__ import annotations

import timm
import torch.nn as nn


def build_backbone(
    name: str = "resnet34",
    pretrained: bool = False,
    in_chans: int = 3,
) -> tuple[nn.Module, int]:
    """Create a 2D vision backbone feature extractor.

    Args:
        name: Name of the architecture (e.g., 'resnet34', 'convnext_nano', 'resnet18').
        pretrained: Whether to load ImageNet pre-trained weights.
        in_chans: Number of input image channels (default 3 for RGB/3-slice slab).

    Returns:
        backbone: nn.Module returning feature embeddings (B, num_features).
        num_features: Integer embedding dimension.
    """
    model = timm.create_model(
        name,
        pretrained=pretrained,
        in_chans=in_chans,
        num_classes=0,  # Strips classification head, outputs pooled embeddings
    )
    dim: int = model.num_features
    return model, dim
