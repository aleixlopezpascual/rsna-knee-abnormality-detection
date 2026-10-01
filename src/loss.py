from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
import torch
from torch import nn

from src.config import TARGET_COLS


def compute_inverse_frequency_class_weights(
    df: pd.DataFrame,
    target_cols: Sequence[str] = TARGET_COLS,
    beta: float = 0.35,
    normalize: bool = True,
) -> torch.Tensor:
    """Compute smooth inverse-frequency class weights for multi-label loss balancing.

    weight_c = (N / (pos_c + epsilon)) ** beta

    Args:
        df: Training DataFrame containing target columns.
        target_cols: List of 12 target column names.
        beta: Power exponent (0.0 = uniform, 1.0 = strict inverse frequency, default 0.35).
        normalize: Whether to normalize weights so the mean across classes is 1.0.

    Returns:
        1D Tensor of shape (len(target_cols),).
    """
    n_samples = len(df)
    weights = []

    for col in target_cols:
        if col in df.columns:
            pos_count = float((df[col].fillna(0) >= 0.5).sum())
        else:
            pos_count = n_samples * 0.20  # Default 20% prevalence fallback

        pos_count = max(pos_count, 1.0)
        w = (n_samples / pos_count) ** beta
        weights.append(w)

    w_arr = np.asarray(weights, dtype=np.float32)
    if normalize and len(w_arr) > 0:
        w_arr = w_arr / w_arr.mean()

    return torch.from_numpy(w_arr)


class AsymmetricFocalLoss(nn.Module):
    """Asymmetric Focal Loss (ASL) for multi-label imbalance with probability margin shifting.

    Reference:
        Ridnik et al., "Asymmetric Loss For Multi-Label Classification", ICCV 2021.

    Args:
        gamma_neg: Focusing parameter for negative samples (higher suppresses easy negatives).
        gamma_pos: Focusing parameter for positive samples (typically 0 or 1).
        clip: Probability margin shift on negative samples (discards very easy negatives).
        eps: Small epsilon for log numerical stability.
        class_weights: Optional 1D Tensor of shape (n_classes,) for minority class upweighting.
    """

    def __init__(
        self,
        gamma_neg: float = 4.0,
        gamma_pos: float = 1.0,
        clip: float = 0.05,
        eps: float = 1e-8,
        class_weights: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.eps = eps
        if class_weights is not None:
            self.register_buffer("class_weights", class_weights.float())
        else:
            self.class_weights = None

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        sample_weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward loss calculation.

        Args:
            logits: Predicted logits of shape (Batch, Classes).
            targets: Target labels in [0, 1] of shape (Batch, Classes).
            sample_weights: Optional per-sample confidence weights of shape (Batch, Classes).

        Returns:
            Scalar loss tensor.
        """
        # Compute probabilities
        p = torch.sigmoid(logits)
        xs_pos = p
        xs_neg = 1.0 - p

        # Asymmetric margin clipping for negative instances
        if self.clip is not None and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1.0)

        # Basic log likelihoods
        los_pos = targets * torch.log(xs_pos.clamp(min=self.eps))
        los_neg = (1.0 - targets) * torch.log(xs_neg.clamp(min=self.eps))
        loss = los_pos + los_neg

        # Asymmetric focusing weights
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            pt0 = xs_pos * targets
            pt1 = xs_neg * (1.0 - targets)
            pt = pt0 + pt1
            one_sided_gamma = self.gamma_pos * targets + self.gamma_neg * (1.0 - targets)
            one_sided_w = torch.pow(1.0 - pt, one_sided_gamma)
            loss = loss * one_sided_w

        # Apply minority class weighting
        if self.class_weights is not None:
            loss = loss * self.class_weights.unsqueeze(0)

        # Apply report confidence weights if provided
        if sample_weights is not None:
            loss = loss * sample_weights

        return -loss.mean()
