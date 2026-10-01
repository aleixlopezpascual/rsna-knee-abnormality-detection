from __future__ import annotations

import argparse
from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd

from src.config import TARGET_COLS, get_target_routing_weights
from src.ensemble import rank_normalize_predictions
from src.metrics import compute_target_roc_auc


def optimize_single_target_weight(
    y_true: np.ndarray,
    pred_a: np.ndarray,
    pred_b: np.ndarray,
    n_steps: int = 51,
) -> tuple[float, float, float, float]:
    """Find the optimal blending weight w in [0, 1] for a single target.

    Objective: w * pred_a + (1 - w) * pred_b maximizing ROC-AUC.

    Args:
        y_true: Ground truth binary array (N,).
        pred_a: Stream A ranked predictions (N,).
        pred_b: Stream B ranked predictions (N,).
        n_steps: Number of grid evaluation steps (default 51 for step size 0.02).

    Returns:
        best_w: Optimal weight assigned to Stream A.
        auc_a: Baseline AUC of Stream A alone (w=1.0).
        auc_b: Baseline AUC of Stream B alone (w=0.0).
        best_auc: Optimized blended AUC.
    """
    y_true_arr = np.asarray(y_true).ravel()
    pa = np.asarray(pred_a).ravel()
    pb = np.asarray(pred_b).ravel()

    auc_a = compute_target_roc_auc(y_true_arr, pa)
    auc_b = compute_target_roc_auc(y_true_arr, pb)

    best_w = 0.5
    best_auc = max(auc_a, auc_b)

    for w in np.linspace(0.0, 1.0, n_steps):
        blend = w * pa + (1.0 - w) * pb
        score = compute_target_roc_auc(y_true_arr, blend)
        if score > best_auc:
            best_auc = score
            best_w = float(w)

    # If stream A is strictly best, snap to 1.0; if B is strictly best, snap to 0.0
    if auc_a >= best_auc:
        best_w, best_auc = 1.0, auc_a
    elif auc_b >= best_auc:
        best_w, best_auc = 0.0, auc_b

    return best_w, auc_a, auc_b, best_auc


def optimize_target_routing_weights(
    y_true_df: pd.DataFrame,
    stream_a_df: pd.DataFrame,
    stream_b_df: pd.DataFrame,
    target_cols: Sequence[str] = TARGET_COLS,
    id_col: str = "StudyInstanceUID",
    n_steps: int = 51,
) -> dict[str, Any]:
    """Optimize continuous blending weights between two model streams across all 12 targets.

    Args:
        y_true_df: Ground truth DataFrame.
        stream_a_df: Stream A (e.g. CNN/CoAtNet) predictions.
        stream_b_df: Stream B (e.g. Transformer/DINOv2) predictions.
        target_cols: Target column names.
        id_col: Study UID column name.
        n_steps: Grid search steps per target.

    Returns:
        Dictionary containing optimal weights, per-target AUC breakdowns, and macro summary.
    """
    targets = list(target_cols)

    # Align on ID
    y_aligned = y_true_df.set_index(id_col)
    common_ids = [uid for uid in y_aligned.index if uid in stream_a_df[id_col].values and uid in stream_b_df[id_col].values]

    if not common_ids:
        # Fallback to positional alignment if mock or unindexed
        n_common = min(len(y_true_df), len(stream_a_df), len(stream_b_df))
        y_sub = y_true_df.iloc[:n_common]
        a_sub = stream_a_df.iloc[:n_common]
        b_sub = stream_b_df.iloc[:n_common]
    else:
        y_sub = y_true_df[y_true_df[id_col].isin(common_ids)].set_index(id_col).reindex(common_ids).reset_index()
        a_sub = stream_a_df[stream_a_df[id_col].isin(common_ids)].set_index(id_col).reindex(common_ids).reset_index()
        b_sub = stream_b_df[stream_b_df[id_col].isin(common_ids)].set_index(id_col).reindex(common_ids).reset_index()

    a_ranked = rank_normalize_predictions(a_sub, targets)
    b_ranked = rank_normalize_predictions(b_sub, targets)

    weights: dict[str, float] = {}
    target_results: dict[str, dict[str, float]] = {}

    macro_a_list = []
    macro_b_list = []
    macro_opt_list = []

    default_weights = get_target_routing_weights()
    macro_default_list = []

    for col in targets:
        y_col = y_sub[col].to_numpy()
        pa = a_ranked[col].to_numpy()
        pb = b_ranked[col].to_numpy()

        best_w, auc_a, auc_b, best_auc = optimize_single_target_weight(
            y_true=y_col, pred_a=pa, pred_b=pb, n_steps=n_steps
        )

        # Baseline default routing score
        w_def = default_weights.get(col, 0.60)
        def_blend = w_def * pa + (1.0 - w_def) * pb
        auc_default = compute_target_roc_auc(y_col, def_blend)

        weights[col] = best_w
        target_results[col] = {
            "weight_a": best_w,
            "auc_stream_a": auc_a,
            "auc_stream_b": auc_b,
            "auc_default_routing": auc_default,
            "auc_optimized_routing": best_auc,
            "delta_vs_stream_a": best_auc - auc_a,
            "delta_vs_stream_b": best_auc - auc_b,
            "delta_vs_default": best_auc - auc_default,
        }

        macro_a_list.append(auc_a)
        macro_b_list.append(auc_b)
        macro_default_list.append(auc_default)
        macro_opt_list.append(best_auc)

    return {
        "weights": weights,
        "target_results": target_results,
        "macro_summary": {
            "macro_auc_stream_a": float(np.mean(macro_a_list)),
            "macro_auc_stream_b": float(np.mean(macro_b_list)),
            "macro_auc_default_routing": float(np.mean(macro_default_list)),
            "macro_auc_optimized_routing": float(np.mean(macro_opt_list)),
            "net_gain_vs_default": float(np.mean(macro_opt_list) - np.mean(macro_default_list)),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Optimize Target-Specific Routing Weights")
    parser.add_argument("--ground-truth", type=str, default="input/train_folds.parquet")
    parser.add_argument("--stream-a", type=str, required=True, help="Predictions from Model Family A")
    parser.add_argument("--stream-b", type=str, required=True, help="Predictions from Model Family B")
    parser.add_argument("--fold", type=int, default=None, help="Optional fold filter (0-4)")
    parser.add_argument("--steps", type=int, default=51, help="Grid search granularity")

    args = parser.parse_args()

    y_df = pd.read_parquet(args.ground_truth)
    if args.fold is not None and "fold" in y_df.columns:
        y_df = y_df[y_df["fold"] == args.fold].reset_index(drop=True)
        print(f"Evaluating exclusively on Validation Fold {args.fold} ({len(y_df)} studies).")

    a_df = pd.read_parquet(args.stream_a) if args.stream_a.endswith(".parquet") else pd.read_csv(args.stream_a)
    b_df = pd.read_parquet(args.stream_b) if args.stream_b.endswith(".parquet") else pd.read_csv(args.stream_b)

    results = optimize_target_routing_weights(y_df, a_df, b_df, n_steps=args.steps)
    summary = results["macro_summary"]

    print("\n" + "=" * 70)
    print("🎯 OPTIMIZED TARGET ROUTING RESULTS")
    print("=" * 70)
    for col, res in results["target_results"].items():
        print(
            f"{col:20s} | Weight A: {res['weight_a']:.2f} | "
            f"A: {res['auc_stream_a']:.4f} | B: {res['auc_stream_b']:.4f} | "
            f"Opt: {res['auc_optimized_routing']:.4f} (Δ {res['delta_vs_default']:+.4f})"
        )
    print("-" * 70)
    print(f"Macro-AUC Stream A Alone:       {summary['macro_auc_stream_a']:.5f}")
    print(f"Macro-AUC Stream B Alone:       {summary['macro_auc_stream_b']:.5f}")
    print(f"Macro-AUC Default Routing:      {summary['macro_auc_default_routing']:.5f}")
    print(f"Macro-AUC OPTIMIZED Routing:    {summary['macro_auc_optimized_routing']:.5f}")
    print(f"Net Gain Over Default Routing:  {summary['net_gain_vs_default']:+.5f} AUC")
    print("=" * 70)


if __name__ == "__main__":
    main()
