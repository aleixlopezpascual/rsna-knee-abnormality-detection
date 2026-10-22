import dataclasses

import numpy as np
import pandas as pd
import pytest

# TDD: This import will fail initially since the implementation does not exist yet.
from src.series_selection import (
    SLOT_NAMES,
    apply_contrast_fallback,
    audit_series_metadata,
    build_candidate_count_table,
    slot_index,
)


def test_slot_names_definition():
    """Verify SLOT_NAMES tuple exactly matches the documented canonical strings in order."""
    assert isinstance(SLOT_NAMES, tuple)
    assert len(SLOT_NAMES) == 6
    assert SLOT_NAMES == (
        "sagittal_fs",
        "coronal_fs",
        "axial_fs",
        "sagittal_nonfs",
        "coronal_nonfs",
        "axial_nonfs",
    )


def test_slot_index_valid_mappings():
    """Verify standard mapping to 0..5 based on plane and fat_suppression flag."""
    # FS planes -> 0, 1, 2
    assert slot_index("Sagittal", 1) == 0
    assert slot_index("Coronal", True) == 1
    assert slot_index("Axial", 1.0) == 2

    # Non-FS planes -> 3, 4, 5
    assert slot_index("Sagittal", 0) == 3
    assert slot_index("Coronal", False) == 4
    assert slot_index("Axial", 0.0) == 5


def test_slot_index_normalization_and_invalid():
    """Verify whitespace/case normalization and rejection of invalid inputs."""
    # Normalization
    assert slot_index(" sagittal ", True) == 0
    assert slot_index("CORONAL", 0) == 4

    # Invalid missing/unknown plane
    assert slot_index(None, 1) is None
    assert slot_index("Unknown", 1) is None

    # Invalid non-binary flag (as per spec: "non-binary/missing flag returns None")
    assert slot_index("Sagittal", None) is None
    assert slot_index("Sagittal", 2) is None
    assert slot_index("Coronal", -1) is None
    assert slot_index("Axial", "1") is None  # strict typing check expectation


def test_slot_index_numpy_scalars():
    """Verify numpy scalar types are accepted for fat_suppression flag."""
    assert slot_index("Sagittal", np.int64(1)) == 0
    assert slot_index("Coronal", np.float32(0)) == 4
    assert slot_index("Axial", np.bool_(True)) == 2


def test_build_candidate_count_table():
    """Verify candidate count aggregation preserves studies and assigns columns correctly."""
    series_df = pd.DataFrame(
        {
            "StudyInstanceUID": ["study1", "study1", "study1", "study2", "study3"],
            "SeriesInstanceUID": ["ser1", "ser2", "ser3", "ser4", "ser5"],
            "Anatomical_Plane": ["Sagittal", "Sagittal", "Coronal", "Unknown", "Axial"],
            "Fat_Suppression": [1, 1, 0, 1, 1],
        }
    )

    counts = build_candidate_count_table(series_df)

    assert counts.index.name == "StudyInstanceUID"
    assert list(counts.columns) == list(SLOT_NAMES)

    # study1: 2 sag_fs, 1 cor_nonfs
    assert counts.loc["study1", "sagittal_fs"] == 2
    assert counts.loc["study1", "coronal_nonfs"] == 1
    assert counts.loc["study1", "axial_fs"] == 0

    # study2: plane unknown -> unmappable, but study should exist with 0 counts
    assert "study2" in counts.index
    assert counts.loc["study2"].sum() == 0

    # study3: 1 ax_fs
    assert counts.loc["study3", "axial_fs"] == 1


def test_build_candidate_count_table_missing_columns():
    """Verify failure on missing required columns."""
    bad_df = pd.DataFrame({"StudyInstanceUID": ["s1"], "Anatomical_Plane": ["Sagittal"]})
    with pytest.raises((KeyError, ValueError, AttributeError)):
        build_candidate_count_table(bad_df)


def test_apply_contrast_fallback():
    """Verify pairwise fallback copies when exactly one side is 0, without mutating input."""
    # Indices correlate to SLOT_NAMES: 0=sag_fs, 1=cor_fs, 2=ax_fs, 3=sag_nonfs, 4=cor_nonfs, 5=ax_nonfs
    counts = pd.DataFrame(
        [
            [0, 1, 0, 0, 0, 2],  # s1: sag(0,0), cor(1,0), ax(0,2)
            [1, 1, 1, 1, 1, 1],  # s2: all present(1,1) -> no change
            [2, 0, 0, 0, 0, 0],  # s3: sag(2,0)
            [0, 0, 0, 0, 3, 0],  # s4: cor(0,3)
        ],
        columns=list(SLOT_NAMES),
        index=["s1", "s2", "s3", "s4"],
    )
    original_counts = counts.copy()

    fallback = apply_contrast_fallback(counts)

    # Ensure input is not mutated
    pd.testing.assert_frame_equal(counts, original_counts)

    # s1 checks
    assert fallback.loc["s1", "sagittal_fs"] == 0
    assert fallback.loc["s1", "sagittal_nonfs"] == 0  # Both zero -> unchanged
    assert fallback.loc["s1", "coronal_fs"] == 1
    assert fallback.loc["s1", "coronal_nonfs"] == 1  # Coronal FS -> NonFS
    assert fallback.loc["s1", "axial_fs"] == 2  # Axial NonFS -> FS
    assert fallback.loc["s1", "axial_nonfs"] == 2

    # s2 checks (all unchanged)
    assert fallback.loc["s2"].sum() == 6

    # s3 & s4 partials
    assert fallback.loc["s3", "sagittal_nonfs"] == 2
    assert fallback.loc["s4", "coronal_fs"] == 3


def test_apply_contrast_fallback_validation():
    """Verify fallback validates input columns."""
    bad_counts = pd.DataFrame({"sagittal_fs": [1], "wrong_col": [0]})
    with pytest.raises((KeyError, ValueError)):
        apply_contrast_fallback(bad_counts)


def test_audit_series_metadata():
    """Verify audit dataclass creation, fields, and correct stat aggregations."""
    df = pd.DataFrame(
        {
            "StudyInstanceUID": ["st1", "st1", "st2"],
            "SeriesInstanceUID": ["se1", "se2", "se3"],
            "Anatomical_Plane": ["Sagittal", "Sagittal", "Coronal"],
            "Fat_Suppression": [1, 0, 1],
            "Fluid_Sensitive": [1, 1, 1],  # For mismatch check
            "n_slices": [30, 25, 40],
        }
    )

    folds_df = pd.DataFrame({"StudyInstanceUID": ["st1", "st2"], "fold": [0, 1]})
    gold_ids = ["st1"]

    audit = audit_series_metadata(df, folds_df=folds_df, gold_ids=gold_ids)

    # Dataclass check
    assert dataclasses.is_dataclass(audit)

    # Core fields
    assert audit.total_rows == 3
    assert audit.mapped_rows == 3
    assert audit.unmapped_rows == 0
    assert audit.unique_studies == 2
    assert audit.has_n_slices is True
    assert audit.fluid_fat_mismatches == 1

    # Folds and Gold coverage
    assert audit.fold_study_counts == {0: 1, 1: 1}
    assert audit.gold_study_count == 1
    assert audit.gold_coverage["sagittal_fs"] == 1  # st1 has sagittal_fs

    # Per-slot field checks
    assert "sagittal_fs" in audit.per_slot
    sag_fs_stats = audit.per_slot["sagittal_fs"]
    assert "raw_rows" in sag_fs_stats
    assert "studies_present" in sag_fs_stats
    assert "multi_candidate_studies" in sag_fs_stats
    assert "studies_after_fallback" in sag_fs_stats
    assert "fallback_count" in sag_fs_stats
    assert "missing_after_fallback" in sag_fs_stats
