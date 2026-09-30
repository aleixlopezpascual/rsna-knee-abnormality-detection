import numpy as np
import pandas as pd
import pytest

from src.config import TARGET_COLS
from src.metrics import (
    compute_target_roc_auc,
    compute_competition_metric,
    compute_detailed_metrics,
)


def test_compute_target_roc_auc_perfect():
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0.1, 0.2, 0.8, 0.9])
    assert compute_target_roc_auc(y_true, y_pred) == 1.0


def test_compute_target_roc_auc_inverted():
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0.9, 0.8, 0.2, 0.1])
    assert compute_target_roc_auc(y_true, y_pred) == 0.0


def test_compute_target_roc_auc_single_class():
    y_true = np.array([0, 0, 0, 0])
    y_pred = np.array([0.1, 0.2, 0.3, 0.4])
    assert compute_target_roc_auc(y_true, y_pred) == 0.5


def test_compute_competition_metric_numpy():
    n_samples = 20
    n_targets = len(TARGET_COLS)
    np.random.seed(42)

    # 12 targets with binary ground truth
    y_true = np.random.randint(0, 2, size=(n_samples, n_targets))
    # Perfect predictions
    y_pred = y_true.astype(float)

    score = compute_competition_metric(y_true, y_pred)
    assert score == pytest.approx(1.0, abs=1e-5)


def test_compute_competition_metric_dataframe():
    n_samples = 20
    np.random.seed(42)

    data_true = {col: np.random.randint(0, 2, size=n_samples) for col in TARGET_COLS}
    df_true = pd.DataFrame(data_true)

    # Predictions with some random noise
    data_pred = {col: np.clip(df_true[col] * 0.8 + 0.1, 0, 1) for col in TARGET_COLS}
    df_pred = pd.DataFrame(data_pred)

    score = compute_competition_metric(df_true, df_pred)
    assert score == pytest.approx(1.0, abs=1e-5)

    detailed = compute_detailed_metrics(df_true, df_pred)
    assert "macro_avg" in detailed
    assert len(detailed) == len(TARGET_COLS) + 1
