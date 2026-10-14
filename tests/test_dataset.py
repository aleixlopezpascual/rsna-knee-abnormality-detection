from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from src.config import TARGET_COLS
from src.dataset import (
    N_SLOTS,
    KneeMultiSlotDataset,
    compute_slice_normal,
    compute_slice_position,
    crop_physical_mm,
    generate_report_grouped_folds,
    infer_laterality_from_geometry,
    normalise_laterality,
)


def test_slice_geometry_axial():
    # Standard Axial orientation: row along +x (1, 0, 0), col along +y (0, 1, 0)
    iop = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
    normal = compute_slice_normal(iop)
    np.testing.assert_allclose(normal, [0.0, 0.0, 1.0], atol=1e-6)

    # Position at z = 45.0 mm
    ipp = [10.0, -20.0, 45.0]
    pos = compute_slice_position(ipp, iop)
    assert pytest.approx(pos, 1e-4) == 45.0


def test_infer_laterality_from_geometry():
    # Right knee: center_x should be negative (patient's right is -x)
    iop = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
    ipp_right = [-228.0, -100.0, 0.0]
    pixel_spacing = [0.5, 0.5]
    lat_r = infer_laterality_from_geometry(ipp_right, iop, pixel_spacing, rows=512, cols=512)
    assert lat_r == "R"

    # Left knee: center_x should be positive (patient's left is +x)
    ipp_left = [20.0, -100.0, 0.0]
    lat_l = infer_laterality_from_geometry(ipp_left, iop, pixel_spacing, rows=512, cols=512)
    assert lat_l == "L"


def test_normalise_laterality_numpy_and_torch():
    # 3-slice slab (3, 4, 4)
    arr = np.arange(48).reshape(3, 4, 4)
    tensor = torch.from_numpy(arr)

    # For Left knee ('L'), identity
    out_l = normalise_laterality(tensor, plane="Coronal", laterality="L")
    assert torch.equal(out_l, tensor)

    # For Right knee ('R') Coronal, horizontal flip (last dim)
    out_cor_r = normalise_laterality(tensor, plane="Coronal", laterality="R")
    assert torch.equal(out_cor_r, torch.flip(tensor, dims=[-1]))

    # For Right knee ('R') Sagittal, slice reverse (0th dim)
    out_sag_r = normalise_laterality(tensor, plane="Sagittal", laterality="R")
    assert torch.equal(out_sag_r, torch.flip(tensor, dims=[0]))


def test_crop_physical_mm():
    # Image 200x200, pixel spacing 0.5 mm -> physical size 100mm
    # Asking for 130mm crop should return original (want_px 260 > image size 200)
    img = np.ones((200, 200))
    cropped = crop_physical_mm(img, pixel_spacing=0.5, crop_mm=130.0)
    assert cropped.shape == (200, 200)

    # Image 500x500, pixel spacing 0.5 mm -> physical size 250mm
    # Asking for 130mm crop -> want_px 260 < 500 -> cropped to 260x260
    img_large = np.ones((500, 500))
    cropped_large = crop_physical_mm(img_large, pixel_spacing=0.5, crop_mm=130.0)
    assert cropped_large.shape == (260, 260)


def test_generate_report_grouped_folds():
    # Create synthetic dataframe with duplicate reports across 15 distinct report groups
    unique_reports = [f"Report template {i}" for i in range(15)]
    reports = unique_reports * 4  # 60 studies across 15 templates

    df = pd.DataFrame({
        "StudyInstanceUID": [f"study_{i}" for i in range(len(reports))],
        "Report": reports,
        "ACL": ([0.1] * 7 + [0.9] * 8) * 4,
        "Effusion": ([0.9] * 5 + [0.1] * 10) * 4,
    })

    folded_df = generate_report_grouped_folds(df, target_cols=["ACL", "Effusion"], n_splits=5)
    assert "fold" in folded_df.columns
    assert set(folded_df["fold"].unique()) == {0, 1, 2, 3, 4}

    # Group check: every identical report must have exactly one unique fold
    grouped_folds = folded_df.groupby("report_hash")["fold"].nunique()
    assert (grouped_folds == 1).all(), "Leak detected: identical report assigned to multiple folds!"


def test_knee_multi_slot_dataset():
    df = pd.DataFrame({
        "StudyInstanceUID": ["study_01"],
        "ACL": [1.0],
        "ACL__conf": [0.9],
    })

    # Mock one slot present: SAG_FLUID_FS
    mock_slots = {
        "study_01": {
            "SAG_FLUID_FS": np.random.randn(3, 100, 100).astype(np.float32),
        }
    }

    ds = KneeMultiSlotDataset(df, slots_data=mock_slots, img_size=224)
    item = ds[0]

    assert item["study_id"] == "study_01"
    assert item["imgs"].shape == (N_SLOTS, 3, 224, 224)
    assert item["mask"].shape == (N_SLOTS,)
    # Slot 0 is SAG_FLUID_FS, so mask[0] should be 1.0, others 0.0
    assert item["mask"][0] == 1.0
    assert item["mask"][1:].sum() == 0.0
    assert item["targets"].shape == (len(TARGET_COLS),)
    assert item["weights"].shape == (len(TARGET_COLS),)


def test_apply_anatomical_mirror_tta():
    from src.infer import apply_anatomical_mirror_tta

    # Create dummy multi-slot tensor (B=2, S=3, C=3, H=10, W=10)
    imgs = torch.arange(2 * 3 * 3 * 10 * 10, dtype=torch.float32).view(2, 3, 3, 10, 10)
    planes = ["Sagittal", "Coronal", "Axial"]

    mirrored = apply_anatomical_mirror_tta(imgs, planes=planes)
    assert mirrored.shape == imgs.shape

    # Sagittal (slot 0): slice order reversed along channel dimension (C=3)
    assert torch.allclose(mirrored[:, 0, 0], imgs[:, 0, 2])
    assert torch.allclose(mirrored[:, 0, 1], imgs[:, 0, 1])
    assert torch.allclose(mirrored[:, 0, 2], imgs[:, 0, 0])

    # Coronal (slot 1): horizontal flip along width dimension (dim -1)
    assert torch.allclose(mirrored[:, 1, :, :, 0], imgs[:, 1, :, :, -1])

    # Axial (slot 2): horizontal flip along width dimension (dim -1)
    assert torch.allclose(mirrored[:, 2, :, :, 0], imgs[:, 2, :, :, -1])

