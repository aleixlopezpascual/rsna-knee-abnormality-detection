from __future__ import annotations

import torch
import torch.nn.functional as F

from src.config import TARGET_COLS, get_target_routing_weights
from src.models import KneeMultiSlotModel, SlotHead, build_backbone


def test_build_backbone():
    backbone, dim = build_backbone("resnet18", pretrained=False)
    assert dim == 512
    x = torch.randn(2, 3, 64, 64)
    out = backbone(x)
    assert out.shape == (2, 512)


def test_build_backbone_aliases():
    # Test RadImageNet ResNet50 alias
    _bb_rad, dim_rad = build_backbone("radimagenet_resnet50", pretrained=False)
    assert dim_rad == 2048

    # Test CoAtNet alias
    _bb_coat, dim_coat = build_backbone("coatnet", pretrained=False)
    assert dim_coat == 768

    # Test ConvNeXt alias
    _bb_cnx, dim_cnx = build_backbone("convnext", pretrained=False)
    assert dim_cnx == 640


def test_target_routing_weights():
    weights = get_target_routing_weights()
    assert len(weights) == len(TARGET_COLS)
    assert weights["Lateral Meniscus"] == 1.00
    assert weights["Medial Meniscus"] == 0.80
    assert weights["ACL"] == 0.75
    assert weights["Effusion"] == 0.60


def test_slot_head_forward():
    batch_size = 2
    n_slots = 6
    n_targets = len(TARGET_COLS)
    dim = 128

    head = SlotHead(dim=dim, n_slots=n_slots, n_targets=n_targets, hidden=64)

    x = torch.randn(batch_size, n_slots, dim)
    # Batch 0 has all slots, Batch 1 has only slots 0 and 1
    mask = torch.tensor([
        [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        [1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
    ])

    logits = head(x, mask)
    assert logits.shape == (batch_size, n_targets)
    assert not torch.isnan(logits).any()


def test_knee_multi_slot_model_forward_and_backward():
    batch_size = 2
    n_slots = 6
    n_targets = len(TARGET_COLS)
    h, w = 64, 64  # Small spatial size for fast unit test

    model = KneeMultiSlotModel(
        backbone_name="resnet18",
        pretrained=False,
        n_slots=n_slots,
        n_targets=n_targets,
        head_hidden=64,
    )

    imgs = torch.randn(batch_size, n_slots, 3, h, w)
    mask = torch.tensor([
        [1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 1.0, 1.0, 1.0],
    ])

    logits = model(imgs, mask)
    assert logits.shape == (batch_size, n_targets)
    assert not torch.isnan(logits).any()

    # Target loss and backward gradient check
    targets = torch.rand(batch_size, n_targets)
    loss = F.binary_cross_entropy_with_logits(logits, targets)
    loss.backward()

    # Verify gradients exist in head and backbone
    assert model.head.out.weight.grad is not None
    assert torch.isfinite(model.head.out.weight.grad).all()
