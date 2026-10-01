from __future__ import annotations

import argparse
from typing import Any

import pandas as pd

from src.config import TARGET_COLS
from src.metrics import compute_detailed_metrics

MINORITY_TARGETS: list[str] = ["Fracture", "MCL", "Contusion", "Lateral Meniscus"]


def evaluate_specialist_gate(
    baseline_scores: dict[str, float],
    specialist_scores: dict[str, float],
    min_rare_delta: float = 0.02,
) -> tuple[bool, dict[str, float]]:
    """Evaluate whether the specialist model passes the 120% Certainty Gate.

    Criteria:
    - At least +min_rare_delta (+0.02) gain on Fracture
    - At least +min_rare_delta (+0.02) gain on MCL
    - Macro-AUC does not degrade overall (delta >= -0.005)

    Args:
        baseline_scores: Dict of per-target ROC-AUC scores for baseline.
        specialist_scores: Dict of per-target ROC-AUC scores for specialist.
        min_rare_delta: Minimum required AUC improvement on primary rare targets.

    Returns:
        passed: Boolean indicating whether gate criteria are satisfied.
        deltas: Dict of per-target score differences (specialist - baseline).
    """
    deltas = {}
    for k, spec_val in specialist_scores.items():
        if k in baseline_scores:
            deltas[k] = float(spec_val - baseline_scores[k])

    fracture_pass = deltas.get("Fracture", 0.0) >= min_rare_delta
    mcl_pass = deltas.get("MCL", 0.0) >= min_rare_delta
    macro_pass = deltas.get("macro_avg", 0.0) >= -0.005

    passed = bool(fracture_pass and mcl_pass and macro_pass)
    return passed, deltas


def evaluate_oof_files(
    ground_truth_path: str,
    baseline_oof_path: str,
    specialist_oof_path: str,
    id_col: str = "StudyInstanceUID",
) -> dict[str, Any]:
    """Evaluate and compare baseline vs specialist OOF predictions."""
    gt_df = pd.read_parquet(ground_truth_path) if ground_truth_path.endswith(".parquet") else pd.read_csv(ground_truth_path)
    base_df = pd.read_parquet(baseline_oof_path) if baseline_oof_path.endswith(".parquet") else pd.read_csv(baseline_oof_path)
    spec_df = pd.read_parquet(specialist_oof_path) if specialist_oof_path.endswith(".parquet") else pd.read_csv(specialist_oof_path)

    # Align on ID
    common_ids = [uid for uid in gt_df[id_col] if uid in base_df[id_col].values and uid in spec_df[id_col].values]
    gt_sub = gt_df[gt_df[id_col].isin(common_ids)].set_index(id_col).reindex(common_ids).reset_index()
    base_sub = base_df[base_df[id_col].isin(common_ids)].set_index(id_col).reindex(common_ids).reset_index()
    spec_sub = spec_df[spec_df[id_col].isin(common_ids)].set_index(id_col).reindex(common_ids).reset_index()

    base_scores = compute_detailed_metrics(gt_sub[TARGET_COLS], base_sub[TARGET_COLS])
    spec_scores = compute_detailed_metrics(gt_sub[TARGET_COLS], spec_sub[TARGET_COLS])

    passed, deltas = evaluate_specialist_gate(base_scores, spec_scores)

    return {
        "gate_passed": passed,
        "baseline_scores": base_scores,
        "specialist_scores": spec_scores,
        "deltas": deltas,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Specialist Model OOF Validation Gate")
    parser.add_argument("--ground-truth", type=str, default="input/train_folds.parquet")
    parser.add_argument("--baseline-oof", type=str, required=True)
    parser.add_argument("--specialist-oof", type=str, required=True)

    args = parser.parse_args()

    results = evaluate_oof_files(args.ground_truth, args.baseline_oof, args.specialist_oof)

    print("\n" + "=" * 70)
    print("🔬 SPECIALIST MODEL VALIDATION GATE EVALUATION")
    print("=" * 70)
    for col in TARGET_COLS + ["macro_avg"]:
        b_score = results["baseline_scores"].get(col, 0.0)
        s_score = results["specialist_scores"].get(col, 0.0)
        delta = results["deltas"].get(col, 0.0)
        marker = "⭐" if col in MINORITY_TARGETS else " "
        print(f"{marker} {col:20s} | Base: {b_score:.4f} | Spec: {s_score:.4f} | Δ {delta:+.4f}")
    print("-" * 70)
    status_str = "✅ PASSED (Go for Full 5-Fold Training)" if results["gate_passed"] else "❌ FAILED (Refine Specialist Loss/Architecture)"
    print(f"Gate Decision: {status_str}")
    print("=" * 70)


if __name__ == "__main__":
    main()
