from __future__ import annotations

import numpy as np

from src.landmark import extract_anatomical_slabs, locate_joint_center_slice


def test_locate_joint_center_slice():
    positions = list(np.linspace(-80.0, 80.0, 32))  # Continuous physical coordinates
    center_idx = locate_joint_center_slice(positions, plane="Coronal")
    assert 14 <= center_idx <= 17  # Centered on physical 0 mm midline


def test_extract_anatomical_slabs():
    vol = np.arange(32 * 64 * 64, dtype=np.float32).reshape(32, 64, 64)
    positions = list(np.linspace(-80.0, 80.0, 32))
    slabs = extract_anatomical_slabs(vol, positions, k_slices=3)
    assert slabs.shape == (3, 64, 64)
    # The middle slice of the slab should correspond to the center slice
    center_idx = locate_joint_center_slice(positions, plane="Coronal")
    np.testing.assert_array_equal(slabs[1], vol[center_idx])
