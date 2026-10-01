from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src.config import TARGET_COLS


def compute_target_roc_auc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute ROC-AUC for a single target, safely handling degenerate single-class batches."""
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()

    valid = ~np.isnan(y_true) & ~np.isnan(y_pred)
    y_true = y_true[valid]
    y_pred = y_pred[valid]

    # If continuous soft labels are passed as y_true, binarize at 0.5
    unique_vals = np.unique(y_true)
    if len(unique_vals) > 2 or not np.all(np.isin(unique_vals, [0, 1])):
        y_true_bin = (y_true >= 0.5).astype(int)
    else:
        y_true_bin = y_true.astype(int)

    unique_classes = np.unique(y_true_bin)
    if len(unique_classes) < 2:
        return 0.5

    return float(roc_auc_score(y_true_bin, y_pred))


def compute_competition_metric(
    y_true: np.ndarray | pd.DataFrame,
    y_pred: np.ndarray | pd.DataFrame,
    target_cols: Sequence[str] = TARGET_COLS,
) -> float:
    """Compute the macro-averaged ROC-AUC across all target columns.

    Args:
        y_true: Ground truth binary targets (N, 12).
        y_pred: Predicted confidence probabilities (N, 12).
        target_cols: List of target column names.

    Returns:
        Macro-averaged ROC-AUC score across all targets.
    """
    if isinstance(y_true, pd.DataFrame):
        y_true_arr = y_true[list(target_cols)].to_numpy()
    else:
        y_true_arr = np.asarray(y_true)

    if isinstance(y_pred, pd.DataFrame):
        y_pred_arr = y_pred[list(target_cols)].to_numpy()
    else:
        y_pred_arr = np.asarray(y_pred)

    num_targets = len(target_cols)
    scores = []
    for i in range(num_targets):
        score = compute_target_roc_auc(y_true_arr[:, i], y_pred_arr[:, i])
        scores.append(score)

    return float(np.mean(scores))


def compute_detailed_metrics(
    y_true: np.ndarray | pd.DataFrame,
    y_pred: np.ndarray | pd.DataFrame,
    target_cols: Sequence[str] = TARGET_COLS,
) -> dict[str, float]:
    """Compute individual ROC-AUC scores for each target, plus macro average."""
    if isinstance(y_true, pd.DataFrame):
        y_true_arr = y_true[list(target_cols)].to_numpy()
    else:
        y_true_arr = np.asarray(y_true)

    if isinstance(y_pred, pd.DataFrame):
        y_pred_arr = y_pred[list(target_cols)].to_numpy()
    else:
        y_pred_arr = np.asarray(y_pred)

    result = {}
    scores = []
    for i, col in enumerate(target_cols):
        score = compute_target_roc_auc(y_true_arr[:, i], y_pred_arr[:, i])
        result[col] = score
        scores.append(score)

    result["macro_avg"] = float(np.mean(scores))
    return result
