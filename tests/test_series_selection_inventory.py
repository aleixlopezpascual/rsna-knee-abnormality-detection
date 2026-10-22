"""Test the private Kaggle series inventory and selection-manifest helpers."""

from pathlib import Path

import pandas as pd
import pytest
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, SecondaryCaptureImageStorage, generate_uid

from notebooks.experiments.series_selection_inventory.series_selection_inventory import (
    SLOT_NAMES,
    build_selection_manifest,
    count_dicom_files,
    scan_series,
    slot_name,
)


def test_slot_names_exact_order():
    """Verify SLOT_NAMES matches exactly the Exp-18 slot order."""
    expected = [
        "sagittal_fs",
        "coronal_fs",
        "axial_fs",
        "sagittal_nonfs",
        "coronal_nonfs",
        "axial_nonfs",
    ]
    assert SLOT_NAMES == expected


def test_slot_name_mapping():
    """Verify normalizations and rejections for the plane x fat_suppression mapping."""
    # Exact matches
    assert slot_name("sagittal", 1) == "sagittal_fs"
    assert slot_name("coronal", 0) == "coronal_nonfs"

    # Case and whitespace normalization
    assert slot_name(" Sagittal ", 1) == "sagittal_fs"
    assert slot_name("AXIAL", 0) == "axial_nonfs"

    # Invalid combinations or missing
    assert slot_name("axial", 2) is None
    assert slot_name("unknown", 1) is None
    assert slot_name("coronal", None) is None
    assert slot_name(None, 1) is None


def test_count_dicom_files(tmp_path: Path):
    """Verify .dcm counting is case-insensitive, non-recursive, and handles missing dirs."""
    series_dir = tmp_path / "1.2.3.series"

    # Directory absent
    assert count_dicom_files(series_dir) == 0

    series_dir.mkdir()

    # Directory empty
    assert count_dicom_files(series_dir) == 0

    # Mixed files
    (series_dir / "1.dcm").touch()
    (series_dir / "2.DCM").touch()
    (series_dir / "3.txt").touch()

    # Subdirectory with .dcm (should not be counted)
    subdir = series_dir / "subdir"
    subdir.mkdir()
    (subdir / "4.dcm").touch()

    assert count_dicom_files(series_dir) == 2


def test_scan_series_inventory(tmp_path: Path):
    """Verify series scanning logic, ensuring descriptor preservation and feature augmentation."""
    data_root = tmp_path

    s_dir1 = data_root / "train_series" / "study1" / "series1"
    s_dir1.mkdir(parents=True)
    (s_dir1 / "1.dcm").touch()
    (s_dir1 / "2.dcm").touch()

    # series2 directory is intentionally absent

    descriptors = pd.DataFrame(
        {
            "StudyInstanceUID": ["study1", "study1"],
            "SeriesInstanceUID": ["series1", "series2"],
            "SeriesDescription": ["Sagittal T2 FS", "Axial T1"],
            "Anatomical_Plane": ["Sagittal", "Axial"],
            "Fat_Suppression": [1, 0],
        }
    )

    res = scan_series(data_root, descriptors, read_headers=False)

    assert "slot_name" in res.columns
    assert "series_dir_exists" in res.columns
    assert "n_slices" in res.columns

    # Preserves original columns
    assert "SeriesDescription" in res.columns

    # First series (present)
    row1 = res[res["SeriesInstanceUID"] == "series1"].iloc[0]
    assert row1["slot_name"] == "sagittal_fs"
    assert bool(row1["series_dir_exists"])
    assert row1["n_slices"] == 2

    # Second series (absent)
    row2 = res[res["SeriesInstanceUID"] == "series2"].iloc[0]
    assert row2["slot_name"] == "axial_nonfs"
    assert not bool(row2["series_dir_exists"])
    assert row2["n_slices"] == 0


def test_build_selection_manifest_basic_and_fallback():
    """
    Verify exclusion (<3 slices), deterministic ranking, top-two marking,
    and paired fallback logic.
    """
    inventory = pd.DataFrame(
        {
            "StudyInstanceUID": ["s1", "s1", "s1", "s1", "s2", "s2"],
            "SeriesInstanceUID": ["ser1", "ser2", "ser3", "ser4", "ser_fs", "ser_nonfs"],
            "slot_name": [
                "sagittal_fs",  # s1: n_slices=2 (ignored)
                "sagittal_fs",  # s1: n_slices=10 (tie)
                "sagittal_fs",  # s1: n_slices=10 (tie)
                "sagittal_fs",  # s1: n_slices=8
                "coronal_fs",  # s2: n_slices=20
                "sagittal_nonfs",  # s2: n_slices=15 (used as fallback for missing sagittal_fs)
            ],
            "n_slices": [2, 10, 10, 8, 20, 15],
        }
    )

    manifest = build_selection_manifest(inventory)

    # 1. Exclusion rule (n_slices < 3)
    assert "ser1" not in manifest["SeriesInstanceUID"].values

    # 2. Deterministic ranking and top-two marking
    s1_sag_fs = manifest[
        (manifest["StudyInstanceUID"] == "s1") & (manifest["slot_name"] == "sagittal_fs")
    ]
    s1_sag_fs = s1_sag_fs.sort_values("rank_n_slices")

    # Deterministic string ascending sort for tie break between ser2 and ser3
    assert s1_sag_fs.iloc[0]["SeriesInstanceUID"] == "ser2"
    assert bool(s1_sag_fs.iloc[0]["is_baseline_selection"])
    assert bool(s1_sag_fs.iloc[0]["is_top_two"])

    assert s1_sag_fs.iloc[1]["SeriesInstanceUID"] == "ser3"
    assert not bool(s1_sag_fs.iloc[1]["is_baseline_selection"])
    assert bool(s1_sag_fs.iloc[1]["is_top_two"])

    # ser4 is rank 3, not in top two
    assert s1_sag_fs.iloc[2]["SeriesInstanceUID"] == "ser4"
    assert not bool(s1_sag_fs.iloc[2]["is_top_two"])

    # 3. Fallback logic: s2 lacks sagittal_fs, but has sagittal_nonfs
    fallback_row = manifest[
        (manifest["StudyInstanceUID"] == "s2") & (manifest["slot_name"] == "sagittal_fs")
    ]
    assert len(fallback_row) == 1
    assert bool(fallback_row.iloc[0]["is_fallback"])
    assert fallback_row.iloc[0]["source_slot"] == "sagittal_nonfs"
    assert fallback_row.iloc[0]["SeriesInstanceUID"] == "ser_nonfs"
    assert bool(fallback_row.iloc[0]["is_baseline_selection"])


def test_build_selection_manifest_proxies():
    """Verify that proxy metrics trigger additional rankings when present."""
    inventory = pd.DataFrame(
        {
            "StudyInstanceUID": ["s1", "s1", "s1"],
            "SeriesInstanceUID": ["ser1", "ser2", "ser3"],
            "slot_name": ["axial_fs", "axial_fs", "axial_fs"],
            "n_slices": [15, 15, 15],
            "approx_coverage_mm": [100.0, 120.0, 90.0],
            "pixel_area_mm2": [0.4, 0.3, 0.5],
        }
    )

    manifest = build_selection_manifest(inventory)

    assert "rank_coverage" in manifest.columns
    assert "rank_resolution" in manifest.columns

    # ser2 has highest coverage (120.0) -> rank 1
    assert manifest.loc[manifest["SeriesInstanceUID"] == "ser2", "rank_coverage"].iloc[0] == 1

    # ser2 has smallest pixel area (highest resolution, 0.3) -> rank 1
    assert manifest.loc[manifest["SeriesInstanceUID"] == "ser2", "rank_resolution"].iloc[0] == 1


def test_scan_series_reads_header_and_leaves_missing_coverage_unranked(tmp_path: Path):
    """Read only DICOM headers; do not invent coverage when spacing is absent."""

    def write_header(path: Path, with_spacing: bool) -> None:
        file_meta = FileMetaDataset()
        file_meta.MediaStorageSOPClassUID = SecondaryCaptureImageStorage
        file_meta.MediaStorageSOPInstanceUID = generate_uid()
        file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
        dataset = FileDataset(str(path), {}, file_meta=file_meta, preamble=b"\0" * 128)
        dataset.SOPClassUID = file_meta.MediaStorageSOPClassUID
        dataset.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
        dataset.Rows = 64
        dataset.Columns = 64
        dataset.PixelSpacing = [0.5, 0.6]
        if with_spacing:
            dataset.SliceThickness = 2
            dataset.SpacingBetweenSlices = 3
        dataset.save_as(path)

    data_root = tmp_path
    with_spacing = data_root / "train_series" / "study1" / "with_spacing"
    no_spacing = data_root / "train_series" / "study1" / "no_spacing"
    with_spacing.mkdir(parents=True)
    no_spacing.mkdir(parents=True)
    write_header(with_spacing / "1.dcm", with_spacing=True)
    write_header(no_spacing / "1.dcm", with_spacing=False)

    descriptors = pd.DataFrame(
        {
            "StudyInstanceUID": ["study1", "study1"],
            "SeriesInstanceUID": ["with_spacing", "no_spacing"],
            "Anatomical_Plane": ["Sagittal", "Sagittal"],
            "Fat_Suppression": [1, 1],
        }
    )
    result = scan_series(data_root, descriptors, read_headers=True)
    known = result[result["SeriesInstanceUID"] == "with_spacing"].iloc[0]
    missing = result[result["SeriesInstanceUID"] == "no_spacing"].iloc[0]

    assert bool(known["header_read_success"])
    assert known["pixel_area_mm2"] == pytest.approx(0.3)
    assert known["approx_coverage_mm"] == pytest.approx(3.0)
    assert bool(missing["header_read_success"])
    assert pd.isna(missing["approx_coverage_mm"])
