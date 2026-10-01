from __future__ import annotations

import pandas as pd
import torch

from src.config import TARGET_COLS
from src.loss import AsymmetricFocalLoss, compute_inverse_frequency_class_weights


def test_compute_inverse_frequency_class_weights():
    # Synthetic dataframe where Fracture has 1 positive and Effusion has 9 positives
    df = pd.DataFrame({
        "Fracture": [1.0] + [0.0] * 9,
        "Effusion": [1.0] * 9 + [0.0],
        "ACL": [1.0] * 5 + [0.0] * 5,
    })

    weights = compute_inverse_frequency_class_weights(df, target_cols=["Fracture", "Effusion", "ACL"], beta=0.5)
    assert len(weights) == 3
    # Fracture (rare) must have higher weight than Effusion (common)
    assert weights[0] > weights[1]
    # Mean should be approx 1.0
    assert torch.isclose(weights.mean(), torch.tensor(1.0), atol=1e-4)


def test_asymmetric_focal_loss_forward_and_backward():
    batch_size = 4
    n_classes = len(TARGET_COLS)

    logits = torch.randn(batch_size, n_classes, requires_grad=True)
    targets = (torch.rand(batch_size, n_classes) >= 0.5).float()
    sample_weights = torch.rand(batch_size, n_classes)
    class_weights = torch.ones(n_classes)

    loss_fn = AsymmetricFocalLoss(
        gamma_neg=4.0,
        gamma_pos=1.0,
        clip=0.05,
        class_weights=class_weights,
    )

    loss = loss_fn(logits, targets, sample_weights=sample_weights)
    assert loss.ndim == 0
    assert torch.isfinite(loss)
    assert loss.item() > 0.0

    loss.backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()


def test_asymmetric_loss_suppresses_easy_negatives():
    # Target is all 0 (negatives). Logits are very confident negatives (-10.0)
    logits = torch.full((2, 4), -10.0, requires_grad=True)
    targets = torch.zeros((2, 4))

    loss_fn = AsymmetricFocalLoss(gamma_neg=4.0, clip=0.05)
    loss = loss_fn(logits, targets)

    # Loss on extreme easy negatives should be practically 0
    assert loss.item() < 1e-5
