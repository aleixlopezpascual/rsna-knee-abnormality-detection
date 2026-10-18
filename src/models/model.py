from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import torch
from torch import nn

from src.config import TARGET_COLS
from src.models.backbones import build_backbone
from src.models.slot_head import SlotHead


class KneeMultiSlotModel(nn.Module):
    """End-to-end multi-slot knee classifier.

    Maps (Batch, N_SLOTS, 3, H, W) and (Batch, N_SLOTS) presence mask
    to (Batch, 12) abnormality prediction logits.

    Args:
        backbone_name: Model architecture name (default 'resnet34').
        pretrained: Whether to use pretrained weights.
        weights_path: Optional explicit path to custom weights (e.g. RadImageNet .pt).
        n_slots: Number of canonical acquisition slots (default 6).
        n_targets: Number of abnormality findings to predict (default 12).
        head_hidden: Hidden dimension for SlotHead attention.
        dropout: Head dropout probability.
        prior: Whether to apply anatomical clinical prior bias.
        target_cols: List of target column names.
    """

    def __init__(
        self,
        backbone_name: str = "resnet34",
        pretrained: bool = False,
        weights_path: str | Path | None = None,
        n_slots: int = 6,
        n_targets: int = 12,
        head_hidden: int = 256,
        dropout: float = 0.2,
        prior: bool = True,
        target_cols: Sequence[str] = TARGET_COLS,
        img_size: int = 224,
    ) -> None:
        super().__init__()
        self.n_slots = n_slots
        self.n_targets = n_targets

        self.backbone, self.dim = build_backbone(
            name=backbone_name,
            pretrained=pretrained,
            in_chans=3,
            weights_path=weights_path,
            img_size=img_size,
        )

        self.head = SlotHead(
            dim=self.dim,
            n_slots=n_slots,
            n_targets=n_targets,
            hidden=head_hidden,
            dropout=dropout,
            prior=prior,
            target_cols=target_cols,
        )

        # Register ImageNet normalization buffers
        self.register_buffer(
            "mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 1, 3, 1, 1)
        )
        self.register_buffer(
            "std", torch.tensor([0.229, 0.224, 0.225]).view(1, 1, 3, 1, 1)
        )

    def forward(self, imgs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            imgs: Tensor of shape (Batch, Slots, 3, H, W).
            mask: Tensor of shape (Batch, Slots) with 1 for present, 0 for missing.

        Returns:
            logits: Tensor of shape (Batch, 12).
        """
        b, s, c, h, w = imgs.shape

        # Normalize pixel values
        if imgs.max() > 1.0:
            x = imgs / 255.0
        else:
            x = imgs
        x = (x - self.mean) / self.std

        # Flatten slots into batch dimension for efficient 2D backbone forward
        x = x.view(b * s, c, h, w)
        feat = self.backbone(x)  # (b * s, dim)

        # Restore slot structure
        feat = feat.view(b, s, self.dim)

        # Aggregate through slot attention head with mask
        logits = self.head(feat, mask)  # (b, n_targets)
        return logits
