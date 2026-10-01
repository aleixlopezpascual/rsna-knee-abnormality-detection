from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn

from src.config import TARGET_COLS

# Prior preference table mapping 12 targets to preferred slot indices (0 to 5)
# 0: SAG_FLUID_FS, 1: COR_FLUID_FS, 2: AX_FLUID_FS
# 3: SAG_FLUID_NOFS, 4: COR_T1, 5: SAG_T1
SLOT_PRIOR_TABLE: dict[str, tuple[int, ...]] = {
    "ACL": (0, 3, 5),
    "MCL": (1, 4),
    "Medial Meniscus": (0, 1, 3, 4),
    "Lateral Meniscus": (0, 1, 3, 4),
    "Medial OA": (1, 4, 5),
    "Lateral OA": (1, 4, 5),
    "PF OA": (0, 2, 5),
    "Effusion": (0, 2),
    "Synovitis": (0, 2),
    "Baker's": (0,),
    "Contusion": (0, 1, 2),
    "Fracture": (0, 1, 2, 4, 5),
}

SLOT_PRIOR_STRENGTH: float = 0.55


class SlotHead(nn.Module):
    """Per-diagnosis multi-slot attention aggregator with anatomical inductive priors.

    Args:
        dim: Feature embedding dimension per slot.
        n_slots: Total number of slots (default 6).
        n_targets: Total number of targets (default 12).
        hidden: Hidden projection dimension.
        dropout: Dropout rate.
        prior: Whether to apply anatomical logit bias prior.
        target_cols: List of target names.
    """

    def __init__(
        self,
        dim: int,
        n_slots: int = 6,
        n_targets: int = 12,
        hidden: int = 256,
        dropout: float = 0.2,
        prior: bool = True,
        target_cols: Sequence[str] = TARGET_COLS,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.n_slots = n_slots
        self.n_targets = n_targets
        self.hidden = hidden
        self.prior = prior

        self.proj = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden),
            nn.GELU(),
        )
        self.slot_emb = nn.Parameter(torch.randn(n_slots, hidden) * 0.02)
        self.query = nn.Parameter(torch.randn(n_targets, hidden) * 0.02)
        self.drop = nn.Dropout(dropout)
        self.out = nn.Linear(hidden, n_targets)

        # Build fixed anatomical bias prior buffer
        prior_matrix = torch.zeros(n_targets, n_slots, dtype=torch.float32)
        if prior:
            for i, tgt in enumerate(target_cols[:n_targets]):
                if tgt in SLOT_PRIOR_TABLE:
                    for s_idx in SLOT_PRIOR_TABLE[tgt]:
                        if s_idx < n_slots:
                            prior_matrix[i, s_idx] = SLOT_PRIOR_STRENGTH

        self.register_buffer("slot_prior", prior_matrix)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Tensor of shape (Batch, Slots, Dim).
            mask: Tensor of shape (Batch, Slots) where 1 indicates present, 0 missing.

        Returns:
            logits: Tensor of shape (Batch, Targets).
        """
        # x: (B, S, D) -> h: (B, S, H)
        h = self.proj(x) + self.slot_emb.unsqueeze(0)

        # Attention scores: (B, Targets, Slots)
        att = torch.einsum("bsh,oh->bos", h, self.query) / (self.hidden**0.5)

        if self.prior:
            att = att + self.slot_prior.unsqueeze(0)

        # Dynamic mask: mask out missing slots (-1e4) so softmax ignores them
        mask_expanded = mask.unsqueeze(1)  # (B, 1, S)
        att = att.masked_fill(mask_expanded < 0.5, -1e4)
        att = torch.softmax(att, dim=-1)

        # Context aggregation: (B, Targets, Hidden)
        ctx = torch.einsum("bos,bsh->boh", att, h)
        ctx = self.drop(ctx)

        # Output projection per target
        # self.out.weight is (Targets, Hidden)
        logits = (ctx * self.out.weight.unsqueeze(0)).sum(dim=-1) + self.out.bias
        return logits
