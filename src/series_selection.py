from __future__ import annotations

import dataclasses

import pandas as pd

SLOT_NAMES = (
    "sagittal_fs",
    "coronal_fs",
    "axial_fs",
    "sagittal_nonfs",
    "coronal_nonfs",
    "axial_nonfs",
)


def slot_index(plane: str | None, fat_suppression: float | bool | None) -> int | None:
    """
    Map anatomical plane and fat suppression to a slot index 0..5.
    Returns None for unmappable or invalid inputs.
    """
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
    except (TypeError, ValueError, OverflowError):
        return None

    is_fs = bool(fat_suppression)

    if plane_norm == "sagittal":
        base = 0
    elif plane_norm == "coronal":
        base = 1
    elif plane_norm == "axial":
        base = 2
    else:
        return None

    return base if is_fs else base + 3


def build_candidate_count_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a table counting the number of candidate series for each slot, per study.
    Preserves studies with zero mappable candidates.
    """
    required = ["StudyInstanceUID", "SeriesInstanceUID", "Anatomical_Plane", "Fat_Suppression"]
    for col in required:
        if col not in df.columns:
            raise KeyError(f"Missing required column: {col}")

    studies = df["StudyInstanceUID"].unique()

    def map_slot(row: pd.Series) -> str | None:
        idx = slot_index(row.get("Anatomical_Plane"), row.get("Fat_Suppression"))
        return SLOT_NAMES[idx] if idx is not None else None

    slot_names = df.apply(map_slot, axis=1)
    valid_mask = slot_names.notna()

    if not valid_mask.any():
        counts = pd.DataFrame(index=pd.Index([], name="StudyInstanceUID"))
    else:
        valid_df = df[valid_mask].copy()
        valid_df["_slot"] = slot_names[valid_mask]
        counts = valid_df.groupby(["StudyInstanceUID", "_slot"]).size().unstack(fill_value=0)

    for col in SLOT_NAMES:
        if col not in counts.columns:
            counts[col] = 0

    counts = counts[list(SLOT_NAMES)]

    missing_studies = set(studies) - set(counts.index)
    if missing_studies:
        missing_df = pd.DataFrame(0, index=list(missing_studies), columns=list(SLOT_NAMES))
        missing_df.index.name = "StudyInstanceUID"
        counts = pd.concat([counts, missing_df])

    counts = counts.sort_index()
    return counts


def apply_contrast_fallback(counts: pd.DataFrame) -> pd.DataFrame:
    """
    Copy candidate counts between FS and Non-FS pairs if exactly one side is zero.
    Does not mutate the input DataFrame.
    """
    for col in SLOT_NAMES:
        if col not in counts.columns:
            raise KeyError(f"Missing required column: {col}")

    res = counts.copy()

    # pairs: (0,3), (1,4), (2,5) corresponding to Sagittal, Coronal, Axial
    for i in range(3):
        fs_col = SLOT_NAMES[i]
        nonfs_col = SLOT_NAMES[i + 3]

        mask_fs_0 = (res[fs_col] == 0) & (res[nonfs_col] > 0)
        mask_nonfs_0 = (res[nonfs_col] == 0) & (res[fs_col] > 0)

        res.loc[mask_fs_0, fs_col] = res.loc[mask_fs_0, nonfs_col]
        res.loc[mask_nonfs_0, nonfs_col] = res.loc[mask_nonfs_0, fs_col]

    return res


@dataclasses.dataclass
class SeriesAudit:
    total_rows: int
    mapped_rows: int
    unmapped_rows: int
    unique_studies: int
    has_n_slices: bool
    fluid_fat_mismatches: int
    fold_study_counts: dict[int, int]
    gold_study_count: int
    gold_coverage: dict[str, int]
    per_slot: dict[str, dict[str, int]]


def audit_series_metadata(
    df: pd.DataFrame,
    folds_df: pd.DataFrame | None = None,
    gold_ids: list[str] | None = None,
) -> SeriesAudit:
    """
    Produce metadata audit metrics for series selection.
    """
    df = df.copy()
    has_n_slices = "n_slices" in df.columns

    total_rows = len(df)
    unique_studies = df["StudyInstanceUID"].nunique()

    def get_slot_idx(r: pd.Series) -> int | None:
        return slot_index(r.get("Anatomical_Plane"), r.get("Fat_Suppression"))

    slot_indices = df.apply(get_slot_idx, axis=1)
    mapped_mask = slot_indices.notna()
    mapped_rows = int(mapped_mask.sum())
    unmapped_rows = total_rows - mapped_rows

    fluid_fat_mismatches = 0
    if "Fluid_Sensitive" in df.columns and "Fat_Suppression" in df.columns:
        valid_mismatch_mask = df["Fluid_Sensitive"].notna() & df["Fat_Suppression"].notna()
        mism = df.loc[valid_mismatch_mask, "Fluid_Sensitive"].astype(float) != df.loc[
            valid_mismatch_mask, "Fat_Suppression"
        ].astype(float)
        fluid_fat_mismatches = int(mism.sum())

    fold_study_counts = {}
    if (
        folds_df is not None
        and "StudyInstanceUID" in folds_df.columns
        and "fold" in folds_df.columns
    ):
        fold_study_counts = folds_df.groupby("fold")["StudyInstanceUID"].nunique().to_dict()

    gold_study_count = 0
    if gold_ids is not None:
        gold_ids_set = set(gold_ids)
        gold_study_count = len(set(df["StudyInstanceUID"].unique()).intersection(gold_ids_set))

    counts = build_candidate_count_table(df)
    counts_fallback = apply_contrast_fallback(counts)

    gold_coverage = {}
    if gold_ids is not None:
        gold_counts = counts_fallback[counts_fallback.index.isin(gold_ids_set)]
        for slot in SLOT_NAMES:
            gold_coverage[slot] = int((gold_counts[slot] > 0).sum())

    per_slot = {}
    if mapped_rows > 0:
        df["_slot"] = slot_indices[mapped_mask].map(lambda x: SLOT_NAMES[int(x)])
    else:
        df["_slot"] = None

    for slot in SLOT_NAMES:
        raw_rows = int((df["_slot"] == slot).sum())
        studies_present = int((counts[slot] > 0).sum())
        multi_candidate_studies = int((counts[slot] > 1).sum())
        studies_after_fallback = int((counts_fallback[slot] > 0).sum())
        fallback_count = studies_after_fallback - studies_present
        missing_after_fallback = unique_studies - studies_after_fallback

        per_slot[slot] = {
            "raw_rows": raw_rows,
            "studies_present": studies_present,
            "multi_candidate_studies": multi_candidate_studies,
            "studies_after_fallback": studies_after_fallback,
            "fallback_count": fallback_count,
            "missing_after_fallback": missing_after_fallback,
        }

    return SeriesAudit(
        total_rows=total_rows,
        mapped_rows=mapped_rows,
        unmapped_rows=unmapped_rows,
        unique_studies=unique_studies,
        has_n_slices=has_n_slices,
        fluid_fat_mismatches=fluid_fat_mismatches,
        fold_study_counts=fold_study_counts,
        gold_study_count=gold_study_count,
        gold_coverage=gold_coverage,
        per_slot=per_slot,
    )
