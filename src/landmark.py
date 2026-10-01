from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def locate_joint_center_slice(
    positions: Sequence[float] | np.ndarray,
    plane: str = "Coronal",
) -> int:
    """Locate the central joint space slice index based on physical coordinates.

    Args:
        positions: Sequence of 1D physical slice coordinates along the plane normal.
        plane: Anatomical orientation ('Coronal', 'Sagittal', 'Axial').

    Returns:
        Integer index of the central joint space slice.
    """
    arr = np.asarray(positions, dtype=np.float64)
    if len(arr) == 0:
        return 0

    # In sorted physical order, the joint center sits at the median position
    median_val = float(np.median(arr))
    center_idx = int(np.argmin(np.abs(arr - median_val)))
    return center_idx


def extract_anatomical_slabs(
    volume: np.ndarray,
    positions: Sequence[float] | np.ndarray,
    k_slices: int = 3,
    stride: int = 1,
) -> np.ndarray:
    """Extract a k-slice slab centered around the physical anatomical joint center.

    Args:
        volume: 3D array of shape (D, H, W).
        positions: 1D physical slice positions (D,).
        k_slices: Number of slices to extract (default 3 for RGB slab).
        stride: Slice interval spacing (default 1).

    Returns:
        Array of shape (k_slices, H, W).
    """
    d = volume.shape[0]
    center_idx = locate_joint_center_slice(positions)

    half = k_slices // 2
    indices = [center_idx + (i - half) * stride for i in range(k_slices)]
    clamped_indices = [max(0, min(d - 1, idx)) for idx in indices]

    return volume[clamped_indices]
