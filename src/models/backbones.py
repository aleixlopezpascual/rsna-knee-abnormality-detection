from __future__ import annotations

from pathlib import Path

import timm
import torch
from torch import nn

ARCH_ALIASES: dict[str, str] = {
    "radimagenet_resnet50": "resnet50",
    "coatnet": "coatnet_rmlp_1_rw_224",
    "dinov2_small": "vit_small_patch14_dinov2",
    "convnext": "convnext_nano",
}


def build_backbone(
    name: str = "resnet34",
    pretrained: bool = False,
    in_chans: int = 3,
    weights_path: str | Path | None = None,
) -> tuple[nn.Module, int]:
    """Create a 2D vision backbone feature extractor.

    Supports ImageNet, RadImageNet, CoAtNet, and DINOv2 architectures.

    Args:
        name: Name or alias of the architecture.
        pretrained: Whether to load ImageNet pre-trained weights from timm.
        in_chans: Number of input image channels (default 3 for RGB/3-slice slab).
        weights_path: Optional explicit path to custom weights (e.g. RadImageNet .pt).

    Returns:
        backbone: nn.Module returning feature embeddings (B, num_features).
        num_features: Integer embedding dimension.
    """
    actual_name = ARCH_ALIASES.get(name.lower(), name)

    # If loading custom weights from file, avoid downloading generic ImageNet weights
    use_timm_pretrained = pretrained and (weights_path is None)

    model = timm.create_model(
        actual_name,
        pretrained=use_timm_pretrained,
        in_chans=in_chans,
        num_classes=0,  # Strips classification head, outputs pooled embeddings
    )

    if weights_path is not None:
        p = Path(weights_path)
        if p.exists():
            state_dict = torch.load(p, map_location="cpu")
            if "state_dict" in state_dict:
                state_dict = state_dict["state_dict"]
            elif "model" in state_dict:
                state_dict = state_dict["model"]

            # Filter out head parameters and strip 'module.' or 'backbone.' prefix if present
            clean_state = {}
            for k, v in state_dict.items():
                clean_k = k.removeprefix("module.").removeprefix("backbone.")
                if (
                    clean_k in model.state_dict()
                    and model.state_dict()[clean_k].shape == v.shape
                ):
                    clean_state[clean_k] = v

            model.load_state_dict(clean_state, strict=False)

    dim: int = model.num_features
    return model, dim
