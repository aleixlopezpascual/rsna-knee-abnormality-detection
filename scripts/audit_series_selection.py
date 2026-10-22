import argparse
import dataclasses
import json
import sys
from pathlib import Path

import pandas as pd

from src.config import TARGET_COLS
from src.series_selection import audit_series_metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit series selection metadata")
    parser.add_argument(
        "--series", type=str, default="input/train_series.csv", help="Path to train_series.csv"
    )
    parser.add_argument(
        "--folds", type=str, default="input/train_folds.parquet", help="Path to train_folds.parquet"
    )
    parser.add_argument("--train", type=str, default="input/train.csv", help="Path to train.csv")

    args = parser.parse_args()

    series_path = Path(args.series)
    folds_path = Path(args.folds)
    train_path = Path(args.train)

    if not series_path.exists():
        print(json.dumps({"error": f"Series file not found: {series_path}"}))
        sys.exit(1)

    df = pd.read_csv(series_path)

    folds_df = None
    if folds_path.exists():
        folds_df = pd.read_parquet(folds_path)

    gold_ids = None
    if train_path.exists():
        train_df = pd.read_csv(train_path)
        # Detect human-gold rows by all 12 target labels being non-null
        if all(c in train_df.columns for c in TARGET_COLS):
            gold_mask = train_df[TARGET_COLS].notna().all(axis=1)
            gold_ids = train_df.loc[gold_mask, "StudyInstanceUID"].tolist()

    audit = audit_series_metadata(df, folds_df=folds_df, gold_ids=gold_ids)

    result = dataclasses.asdict(audit)
    result["caveat"] = (
        "METADATA-ONLY AUDIT. train_series.csv has no per-series n_slices/pixel-quality "
        "fields and the local input/train_series DICOM directory is absent. Thus, the tool "
        "cannot reproduce the largest-series choice or measure model AUC. "
        "Current Exp-18 policy: Six plane x fat-suppression slots. Series with fewer "
        "than 3 slices are skipped. Remaining candidates are sorted by descending slice "
        "count, inference chooses the first, and missing contrast variants are copied "
        "from their pair. Gold coverage reported is computed AFTER contrast fallback."
    )

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
