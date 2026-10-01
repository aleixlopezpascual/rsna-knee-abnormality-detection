from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import TARGET_COLS
from src.optimize_routing import (
    optimize_single_target_weight,
    optimize_target_routing_weights,
)


def test_optimize_single_target_weight():
    # Synthetic target where Stream A is strongly predictive and Stream B is random noise
    np.random.seed(42)
    y = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    pred_a = np.array([0.1, 0.2, 0.15, 0.3, 0.25, 0.7, 0.8, 0.85, 0.9, 0.95])
    pred_b = np.random.uniform(0, 1, size=10)

    best_w, auc_a, auc_b, best_auc = optimize_single_target_weight(y, pred_a, pred_b, n_steps=21)
    assert best_w >= 0.8
    assert best_auc >= auc_a
    assert best_auc > auc_b


def test_optimize_target_routing_weights():
    studies = [f"s_{i}" for i in range(20)]
    np.random.seed(42)

    y_data = {"StudyInstanceUID": studies}
    a_data = {"StudyInstanceUID": studies}
    b_data = {"StudyInstanceUID": studies}

    for col in TARGET_COLS:
        y_data[col] = (np.random.uniform(0, 1, size=20) >= 0.5).astype(int)
        a_data[col] = np.random.uniform(0, 1, size=20)
        b_data[col] = np.random.uniform(0, 1, size=20)

    y_df = pd.DataFrame(y_data)
    a_df = pd.DataFrame(a_data)
    b_df = pd.DataFrame(b_data)

    res = optimize_target_routing_weights(y_df, a_df, b_df, n_steps=11)

    assert "weights" in res
    assert "target_results" in res
    assert "macro_summary" in res
    assert len(res["weights"]) == len(TARGET_COLS)

    summary = res["macro_summary"]
    assert summary["macro_auc_optimized_routing"] >= summary["macro_auc_stream_a"] - 1e-6
    assert summary["macro_auc_optimized_routing"] >= summary["macro_auc_stream_b"] - 1e-6
