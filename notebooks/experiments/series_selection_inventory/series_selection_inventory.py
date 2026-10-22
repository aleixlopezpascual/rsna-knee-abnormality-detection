from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

SLOT_NAMES = [
    "sagittal_fs",
    "coronal_fs",
    "axial_fs",
    "sagittal_nonfs",
    "coronal_nonfs",
    "axial_nonfs",
]


def slot_name(plane: Any, fat_suppression: Any) -> str | None:
    """Map anatomical plane and fat suppression to a slot name."""
    if not isinstance(plane, str):
        return None
    plane_norm = plane.strip().lower()

    try:
        if isinstance(fat_suppression, str):
            return None
        if pd.isna(fat_suppression):
            return None
        if fat_suppression not in (0, 1):
            return None
    except (TypeError, ValueError):
        return None

    is_fs = bool(fat_suppression)

    if plane_norm == "sagittal":
        return "sagittal_fs" if is_fs else "sagittal_nonfs"
    elif plane_norm == "coronal":
        return "coronal_fs" if is_fs else "coronal_nonfs"
    elif plane_norm == "axial":
        return "axial_fs" if is_fs else "axial_nonfs"
    return None


def count_dicom_files(dir_path: str | Path) -> int:
    """Count .dcm files in a directory (non-recursive, case-insensitive)."""
    dir_path = Path(dir_path)
    if not dir_path.is_dir():
        return 0
    return sum(1 for f in dir_path.iterdir() if f.is_file() and f.suffix.lower() == ".dcm")


def scan_series(
    data_root: str | Path, descriptors: pd.DataFrame, read_headers: bool = False
) -> pd.DataFrame:
    """Scan the series directories, compute slice counts, and optionally extract DICOM metadata."""
    data_root = Path(data_root)
    res = descriptors.copy()

    slots: list[str | None] = []
    dir_exists: list[bool] = []
    n_slices: list[int] = []

    header_features: dict[str, list[Any]] = {}
    if read_headers:
        header_features = {
            "rows": [],
            "columns": [],
            "pixel_spacing_r": [],
            "pixel_spacing_c": [],
            "slice_thickness": [],
            "spacing_between_slices": [],
            "pixel_area_mm2": [],
            "in_plane_fov_r": [],
            "in_plane_fov_c": [],
            "approx_coverage_mm": [],
            "header_read_success": [],
            "header_error": [],
        }

        try:
            import pydicom
            from pydicom.errors import InvalidDicomError
            from pydicom.tag import Tag

            target_tags = [
                Tag("Rows"),
                Tag("Columns"),
                Tag("PixelSpacing"),
                Tag("SliceThickness"),
                Tag("SpacingBetweenSlices"),
            ]
        except ImportError as e:
            raise RuntimeError("pydicom is required when read_headers=True") from e

    for i, (_, row) in enumerate(res.iterrows()):
        if i > 0 and i % 1000 == 0:
            logger.info(f"Scanned {i} series...")

        suid = row.get("StudyInstanceUID")
        seuid = row.get("SeriesInstanceUID")
        plane = row.get("Anatomical_Plane")
        fs = row.get("Fat_Suppression")

        s_name = slot_name(plane, fs)
        slots.append(s_name)

        s_dir = data_root / "train_series" / str(suid) / str(seuid)
        exists = s_dir.is_dir()
        dir_exists.append(exists)

        if not exists:
            n_slices.append(0)
            if read_headers:
                for k in header_features:
                    header_features[k].append(None if k != "header_read_success" else False)
            continue

        dcms = sorted([f for f in s_dir.iterdir() if f.is_file() and f.suffix.lower() == ".dcm"])
        n_slices.append(len(dcms))

        if read_headers:
            if dcms:
                try:
                    ds = pydicom.dcmread(
                        str(dcms[0]), stop_before_pixels=True, specific_tags=target_tags
                    )
                    rows = getattr(ds, "Rows", None)
                    cols = getattr(ds, "Columns", None)
                    ps = getattr(ds, "PixelSpacing", [None, None])
                    ps_r = float(ps[0]) if ps and len(ps) > 0 and ps[0] is not None else None
                    ps_c = float(ps[1]) if ps and len(ps) > 1 and ps[1] is not None else None
                    st_raw = getattr(ds, "SliceThickness", None)
                    sbs_raw = getattr(ds, "SpacingBetweenSlices", None)

                    st = float(st_raw) if st_raw is not None else None
                    if st is not None:
                        st = abs(st)
                        if st == 0:
                            st = None

                    sbs = float(sbs_raw) if sbs_raw is not None else None
                    if sbs is not None:
                        sbs = abs(sbs)
                        if sbs == 0:
                            sbs = None

                    area = ps_r * ps_c if ps_r is not None and ps_c is not None else None
                    fov_r = float(rows) * ps_r if rows is not None and ps_r is not None else None
                    fov_c = float(cols) * ps_c if cols is not None and ps_c is not None else None

                    step = None
                    if sbs is not None:
                        step = sbs
                    elif st is not None:
                        step = st

                    cov = float(len(dcms)) * step if step is not None else None

                    header_features["rows"].append(rows)
                    header_features["columns"].append(cols)
                    header_features["pixel_spacing_r"].append(ps_r)
                    header_features["pixel_spacing_c"].append(ps_c)
                    header_features["slice_thickness"].append(st)
                    header_features["spacing_between_slices"].append(sbs)
                    header_features["pixel_area_mm2"].append(area)
                    header_features["in_plane_fov_r"].append(fov_r)
                    header_features["in_plane_fov_c"].append(fov_c)
                    header_features["approx_coverage_mm"].append(cov)
                    header_features["header_read_success"].append(True)
                    header_features["header_error"].append(None)
                except (InvalidDicomError, OSError, TypeError, ValueError, AttributeError) as e:
                    for k in header_features:
                        if k == "header_read_success":
                            header_features[k].append(False)
                        elif k == "header_error":
                            header_features[k].append(type(e).__name__)
                        else:
                            header_features[k].append(None)
            else:
                for k in header_features:
                    header_features[k].append(None if k != "header_read_success" else False)

    res["slot_name"] = slots
    res["series_dir_exists"] = dir_exists
    res["n_slices"] = n_slices
    if read_headers:
        for k, v in header_features.items():
            res[k] = v

    return res


def apply_contrast_fallback(df: pd.DataFrame) -> pd.DataFrame:
    """Duplicate candidate series between FS and Non-FS pairs if one is empty."""
    df = df.copy()
    if "is_fallback" not in df.columns:
        df["is_fallback"] = False
    if "source_slot" not in df.columns:
        df["source_slot"] = None

    pairs = [
        ("sagittal_fs", "sagittal_nonfs"),
        ("coronal_fs", "coronal_nonfs"),
        ("axial_fs", "axial_nonfs"),
    ]

    fallback_rows: list[pd.DataFrame] = []
    grouped = df.groupby("StudyInstanceUID")
    for _, group in grouped:
        slots_present = set(group["slot_name"].unique())
        for fs_slot, nonfs_slot in pairs:
            if fs_slot in slots_present and nonfs_slot not in slots_present:
                src = group[group["slot_name"] == fs_slot].copy()
                src["slot_name"] = nonfs_slot
                src["is_fallback"] = True
                src["source_slot"] = fs_slot
                fallback_rows.append(src)
            elif nonfs_slot in slots_present and fs_slot not in slots_present:
                src = group[group["slot_name"] == nonfs_slot].copy()
                src["slot_name"] = fs_slot
                src["is_fallback"] = True
                src["source_slot"] = nonfs_slot
                fallback_rows.append(src)

    if fallback_rows:
        return pd.concat([df] + fallback_rows, ignore_index=True)
    return df


def build_selection_manifest(inventory: pd.DataFrame) -> pd.DataFrame:
    """Filter, fallback, and rank candidate series based on slice count and metadata proxies."""
    valid = inventory[inventory["n_slices"] >= 3].copy()

    if len(valid) == 0:
        return pd.DataFrame()

    valid = apply_contrast_fallback(valid)

    valid = valid.sort_values(
        ["StudyInstanceUID", "slot_name", "n_slices", "SeriesInstanceUID"],
        ascending=[True, True, False, True],
    )
    valid["rank_n_slices"] = valid.groupby(["StudyInstanceUID", "slot_name"]).cumcount() + 1
    valid["is_baseline_selection"] = valid["rank_n_slices"] == 1
    valid["is_top_two"] = valid["rank_n_slices"] <= 2

    if "approx_coverage_mm" in valid.columns:
        valid = valid.sort_values(
            ["StudyInstanceUID", "slot_name", "approx_coverage_mm", "SeriesInstanceUID"],
            ascending=[True, True, False, True],
            na_position="last",
        )
        valid["rank_coverage"] = valid.groupby(["StudyInstanceUID", "slot_name"]).cumcount() + 1

    if "pixel_area_mm2" in valid.columns:
        valid = valid.sort_values(
            ["StudyInstanceUID", "slot_name", "pixel_area_mm2", "SeriesInstanceUID"],
            ascending=[True, True, True, True],
            na_position="last",
        )
        valid["rank_resolution"] = valid.groupby(["StudyInstanceUID", "slot_name"]).cumcount() + 1

    valid = valid.sort_values(["StudyInstanceUID", "slot_name", "rank_n_slices"])
    return valid


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate series selection inventory")
    parser.add_argument("--data-root", type=str, default=None, help="Root dir of dataset")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/kaggle/working/series_selection_inventory",
        help="Output directory",
    )
    parser.add_argument("--skip-headers", action="store_true", help="Skip reading DICOM headers")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    data_root: Path | None = None
    if args.data_root:
        data_root = Path(args.data_root)
    else:
        # Bounded search for Kaggle input data
        search_paths = [
            Path("/kaggle/input/rsna-knee-abnormality-detection"),
            Path("/kaggle/input/competitions/rsna-knee-abnormality-detection"),
        ]

        base = Path("/kaggle/input")
        if base.exists():
            search_paths.extend([d for d in base.iterdir() if d.is_dir()])
            for d in base.iterdir():
                if d.is_dir():
                    search_paths.extend([sub_d for sub_d in d.iterdir() if sub_d.is_dir()])

        for p in search_paths:
            if p.is_dir() and (p / "train_series.csv").exists() and (p / "train_series").exists():
                data_root = p
                break

    if not data_root or not data_root.exists():
        logger.error("Could not find competition data root.")
        sys.exit(1)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = data_root / "train_series.csv"
    if not csv_path.exists():
        logger.error(f"Missing {csv_path}")
        sys.exit(1)

    logger.info("Reading train_series.csv...")
    descriptors = pd.read_csv(csv_path)

    read_headers = not args.skip_headers
    logger.info(f"Scanning series... (read_headers={read_headers})")

    inventory = scan_series(data_root, descriptors, read_headers=read_headers)
    inventory.to_csv(out_dir / "series_inventory.csv", index=False)

    logger.info("Building selection manifest...")
    manifest = build_selection_manifest(inventory)
    manifest.to_csv(out_dir / "study_slot_candidates.csv", index=False)

    # Generate summary stats
    total_desc = len(descriptors)
    missing_dirs = int((~inventory["series_dir_exists"]).sum())
    total_slices = int(inventory["n_slices"].sum())
    ineligible = int((inventory["n_slices"] < 3).sum())
    unmappable = int(inventory["slot_name"].isna().sum())

    valid_raw = inventory[inventory["n_slices"] >= 3]
    raw_studies = (
        valid_raw.groupby("slot_name")["StudyInstanceUID"].nunique().to_dict()
        if not valid_raw.empty
        else {}
    )
    post_studies = (
        manifest.groupby("slot_name")["StudyInstanceUID"].nunique().to_dict()
        if not manifest.empty
        else {}
    )

    multi_cand_df = manifest[manifest["rank_n_slices"] > 1]
    multi_candidate = (
        multi_cand_df.groupby("slot_name")["StudyInstanceUID"].nunique().to_dict()
        if not multi_cand_df.empty
        else {}
    )

    failed_headers = 0
    diff_coverage = 0
    diff_resolution = 0

    if read_headers:
        failed_headers = int((~inventory["header_read_success"]).sum())
        if not manifest.empty:
            if "rank_coverage" in manifest.columns:
                valid_cov = manifest[manifest["approx_coverage_mm"].notna()]
                diff_coverage = int(
                    (
                        (valid_cov["is_baseline_selection"] == True)
                        & (valid_cov["rank_coverage"] != 1)
                    ).sum()
                )
            if "rank_resolution" in manifest.columns:
                valid_res = manifest[manifest["pixel_area_mm2"].notna()]
                diff_resolution = int(
                    (
                        (valid_res["is_baseline_selection"] == True)
                        & (valid_res["rank_resolution"] != 1)
                    ).sum()
                )

    summary = {
        "disclaimer": "This is inventory/proxy evidence only and does not evaluate AUC.",
        "total_descriptors": total_desc,
        "missing_directories": missing_dirs,
        "total_dicom_slices": total_slices,
        "ineligible_series_less_than_3_slices": ineligible,
        "unmappable_rows": unmappable,
        "raw_slot_study_coverage": raw_studies,
        "post_fallback_slot_study_coverage": post_studies,
        "multi_candidate_slots": multi_candidate,
        "missing_or_failed_header_reads": failed_headers,
        "baseline_differs_from_coverage_proxy": diff_coverage,
        "baseline_differs_from_resolution_proxy": diff_resolution,
    }

    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=4)

    logger.info("Inventory complete.")
    logger.info("DISCLAIMER: This is inventory/proxy evidence only.")
    logger.info("\n%s", json.dumps(summary, indent=4))


if __name__ == "__main__":
    main()
