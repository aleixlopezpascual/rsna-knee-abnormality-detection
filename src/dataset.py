from __future__ import annotations

import hashlib
from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
import torch
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from torch.utils.data import Dataset

from src.config import SEED, TARGET_COLS

# Canonical 6 slots: 3 anatomical planes x contrast
SLOTS: list[tuple[str, str, bool | None, bool]] = [
    ("SAG_FLUID_FS", "Sagittal", True, True),
    ("COR_FLUID_FS", "Coronal", True, True),
    ("AX_FLUID_FS", "Axial", True, True),
    ("SAG_FLUID_NOFS", "Sagittal", True, False),
    ("COR_T1", "Coronal", False, False),
    ("SAG_T1", "Sagittal", False, False),
]

SLOT_NAMES: list[str] = [s[0] for s in SLOTS]
N_SLOTS: int = len(SLOTS)


def compute_slice_normal(iop: Sequence[float]) -> np.ndarray:
    """Compute the unit normal vector to the imaging plane from ImageOrientationPatient.

    Args:
        iop: 6 direction cosines [r_x, r_y, r_z, c_x, c_y, c_z].

    Returns:
        3D unit normal vector.
    """
    row_vec = np.asarray(iop[:3], dtype=np.float64)
    col_vec = np.asarray(iop[3:], dtype=np.float64)
    normal = np.cross(row_vec, col_vec)
    norm = np.linalg.norm(normal)
    if norm > 1e-8:
        normal /= norm
    return normal


def compute_slice_position(ipp: Sequence[float], iop: Sequence[float]) -> float:
    """Project ImagePositionPatient onto the plane normal to determine true spatial order.

    Args:
        ipp: 3D slice origin [x, y, z].
        iop: 6 direction cosines.

    Returns:
        Scalar coordinate along the normal axis.
    """
    normal = compute_slice_normal(iop)
    pos = np.asarray(ipp, dtype=np.float64)
    return float(np.dot(pos, normal))


def infer_laterality_from_geometry(
    ipp: Sequence[float],
    iop: Sequence[float],
    pixel_spacing: Sequence[float],
    rows: int,
    cols: int,
    threshold_mm: float = 20.0,
) -> str:
    """Infer right ('R') or left ('L') knee from patient coordinate system geometry.

    In DICOM patient coordinates (+x points to patient's left):
    c = p + r * delta_c * (N_c / 2) + d * delta_r * (N_r / 2)
    If center_x < -threshold: 'R' (patient's right)
    If center_x > +threshold: 'L' (patient's left)
    """
    p = np.asarray(ipp, dtype=np.float64)
    r = np.asarray(iop[:3], dtype=np.float64)
    d = np.asarray(iop[3:], dtype=np.float64)
    dr, dc = float(pixel_spacing[0]), float(pixel_spacing[1])

    centre = p + r * dc * (cols / 2.0) + d * dr * (rows / 2.0)
    cx = centre[0]

    if cx < -threshold_mm:
        return "R"
    if cx > threshold_mm:
        return "L"
    return "UNKNOWN"


def normalise_laterality(
    img: torch.Tensor | np.ndarray,
    plane: str,
    laterality: str,
) -> torch.Tensor | np.ndarray:
    """Normalize knee scans to a standard Left-Knee convention.

    Args:
        img: Array/Tensor of shape (C, H, W) or (H, W).
        plane: 'Sagittal', 'Coronal', or 'Axial'.
        laterality: 'R' or 'L'.

    Returns:
        Canonicalized scan.
    """
    if laterality != "R":
        return img

    is_torch = isinstance(img, torch.Tensor)

    if plane in ("Coronal", "Axial"):
        # Medial-lateral axis is horizontal in Coronal and Axial planes
        if is_torch:
            return torch.flip(img, dims=[-1])
        return np.flip(img, axis=-1)

    if plane == "Sagittal":
        # In Sagittal view, medial-to-lateral progression runs in opposite slice direction
        if is_torch:
            return torch.flip(img, dims=[0])
        return np.flip(img, axis=0)

    return img


def crop_physical_mm(
    img: np.ndarray,
    pixel_spacing: float,
    crop_mm: float = 130.0,
) -> np.ndarray:
    """Center crop an image to a fixed physical millimeter extent.

    Args:
        img: 2D image (H, W) or 3D slab (C, H, W).
        pixel_spacing: Millimeters per pixel.
        crop_mm: Target physical field of view in mm (default 130mm).

    Returns:
        Cropped array.
    """
    if pixel_spacing <= 0 or not np.isfinite(pixel_spacing):
        return img

    want_px = round(crop_mm / pixel_spacing)
    h, w = img.shape[-2], img.shape[-1]

    if 16 < want_px < min(h, w):
        top = max(0, (h - want_px) // 2)
        left = max(0, (w - want_px) // 2)
        if img.ndim == 2:
            return img[top : top + want_px, left : left + want_px]
        return img[..., top : top + want_px, left : left + want_px]

    return img


def generate_report_grouped_folds(
    df: pd.DataFrame,
    target_cols: Sequence[str] = TARGET_COLS,
    n_splits: int = 5,
    seed: int = SEED,
) -> pd.DataFrame:
    """Generate 5-fold splits grouped by report hash with multilabel stratification.

    Ensures studies sharing byte-identical reports are never split across train/val,
    preventing ground truth label leakage.
    """
    df = df.copy()

    # Generate MD5 hash for each report to group duplicate reports
    df["report_hash"] = df["Report"].fillna("").apply(
        lambda r: hashlib.md5(str(r).encode("utf-8")).hexdigest()
    )

    targets = list(target_cols)
    # Binary targets for stratification (threshold at 0.5 if continuous)
    bin_targets = [f"bin_{c}" for c in targets]
    for c in targets:
        if c in df.columns:
            df[f"bin_{c}"] = (df[c].fillna(0) >= 0.5).astype(int)
        else:
            df[f"bin_{c}"] = 0

    # Group by report_hash to aggregate targets per group
    grp_df = df.groupby("report_hash")[bin_targets].max().reset_index()

    actual_splits = min(n_splits, len(grp_df))
    if actual_splits < 2:
        df["fold"] = 0
        df.drop(columns=bin_targets, inplace=True)
        return df

    mskf = MultilabelStratifiedKFold(n_splits=actual_splits, shuffle=True, random_state=seed)
    grp_df["fold"] = -1
    for fold, (_, val_idx) in enumerate(mskf.split(grp_df, grp_df[bin_targets])):
        grp_df.loc[val_idx, "fold"] = fold

    hash_to_fold = dict(zip(grp_df["report_hash"], grp_df["fold"]))
    df["fold"] = df["report_hash"].map(hash_to_fold)

    # Clean up temporary stratification columns
    df.drop(columns=bin_targets, inplace=True)
    return df


class KneeMultiSlotDataset(Dataset):
    """PyTorch Dataset for multi-slot 2.5D knee MRI representations.

    Returns:
        imgs: Tensor of shape (N_SLOTS, 3, H, W)
        mask: Tensor of shape (N_SLOTS,) where 1 indicates slot is present, 0 missing
        targets: Tensor of shape (12,)
        weights: Tensor of shape (12,)
    """

    def __init__(
        self,
        df: pd.DataFrame,
        slots_data: dict[str, dict[str, np.ndarray]] | None = None,
        target_cols: Sequence[str] = TARGET_COLS,
        img_size: int = 224,
        is_train: bool = True,
    ) -> None:
        self.df = df.reset_index(drop=True)
        self.slots_data = slots_data or {}
        self.target_cols = list(target_cols)
        self.img_size = img_size
        self.is_train = is_train

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.df.iloc[idx]
        study_id = str(row["StudyInstanceUID"])

        # Fetch or generate (N_SLOTS, 3, H, W) tensor
        study_slots = self.slots_data.get(study_id, {})

        imgs = torch.zeros((N_SLOTS, 3, self.img_size, self.img_size), dtype=torch.float32)
        mask = torch.zeros(N_SLOTS, dtype=torch.float32)

        for slot_idx, slot_name in enumerate(SLOT_NAMES):
            if slot_name in study_slots:
                arr = study_slots[slot_name]  # (3, H, W) or (H, W)
                tensor = torch.from_numpy(arr).float()
                if tensor.ndim == 2:
                    tensor = tensor.unsqueeze(0).repeat(3, 1, 1)
                elif tensor.shape[0] == 1:
                    tensor = tensor.repeat(3, 1, 1)

                if tensor.shape[-2:] != (self.img_size, self.img_size):
                    tensor = torch.nn.functional.interpolate(
                        tensor.unsqueeze(0),
                        size=(self.img_size, self.img_size),
                        mode="bilinear",
                        align_corners=False,
                    ).squeeze(0)

                imgs[slot_idx] = tensor
                mask[slot_idx] = 1.0

        # Targets and confidence weights
        targets = np.zeros(len(self.target_cols), dtype=np.float32)
        weights = np.ones(len(self.target_cols), dtype=np.float32)

        for i, col in enumerate(self.target_cols):
            if col in row and pd.notnull(row[col]):
                targets[i] = float(row[col])
            else:
                targets[i] = 0.5  # Neutral default

            conf_col = f"{col}__conf"
            if conf_col in row and pd.notnull(row[conf_col]):
                weights[i] = float(row[conf_col])

        return {
            "study_id": study_id,
            "imgs": imgs,
            "mask": mask,
            "targets": torch.tensor(targets, dtype=torch.float32),
            "weights": torch.tensor(weights, dtype=torch.float32),
        }
