from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import TARGET_COLS
from src.ensemble import (
    blend_routed_predictions,
    multi_model_rank_average,
    rank_normalize_predictions,
)


def test_rank_normalize_predictions():
    df = pd.DataFrame({
        "StudyInstanceUID": ["s1", "s2", "s3", "s4"],
        "ACL": [0.1, 0.4, 0.2, 0.9],
        "Effusion": [10.0, 50.0, 20.0, 90.0],
    })

    ranked = rank_normalize_predictions(df, target_cols=["ACL", "Effusion"])
    # Values must be within (0, 1]
    assert (ranked["ACL"] > 0.0).all() and (ranked["ACL"] <= 1.0).all()
    # Rank ordering for ACL: s1 (0.1) < s3 (0.2) < s2 (0.4) < s4 (0.9)
    assert ranked.loc[0, "ACL"] < ranked.loc[2, "ACL"] < ranked.loc[1, "ACL"] < ranked.loc[3, "ACL"]


def test_blend_routed_predictions_lateral_meniscus():
    studies = ["s1", "s2", "s3"]

    # CNN predicts: s1 < s2 < s3 for Lateral Meniscus
    cnn_df = pd.DataFrame({"StudyInstanceUID": studies})
    for c in TARGET_COLS:
        cnn_df[c] = [0.1, 0.5, 0.9]

    # Transformer predicts inverted: s3 < s2 < s1
    trans_df = pd.DataFrame({"StudyInstanceUID": studies})
    for c in TARGET_COLS:
        trans_df[c] = [0.9, 0.5, 0.1]

    blended = blend_routed_predictions(cnn_df, trans_df, target_cols=TARGET_COLS)

    # Lateral Meniscus has 100% CNN weight, so blended must follow CNN ordering: s1 < s2 < s3
    assert blended.loc[0, "Lateral Meniscus"] < blended.loc[1, "Lateral Meniscus"] < blended.loc[2, "Lateral Meniscus"]


def test_multi_model_rank_average():
    studies = ["s1", "s2", "s3"]
    df1 = pd.DataFrame({"StudyInstanceUID": studies, "ACL": [0.2, 0.5, 0.8]})
    df2 = pd.DataFrame({"StudyInstanceUID": studies, "ACL": [0.3, 0.6, 0.9]})

    ens = multi_model_rank_average([df1, df2], weights=[0.5, 0.5], target_cols=["ACL"])
    assert len(ens) == 3
    assert "ACL" in ens.columns
    assert (ens["ACL"].to_numpy() == np.array([1/3, 2/3, 1.0])).all()


def test_foundation_specialist_fusion():
    studies = ["s1", "s2", "s3"]
    pub_df = pd.DataFrame({"StudyInstanceUID": studies})
    spec_df = pd.DataFrame({"StudyInstanceUID": studies})
    for c in TARGET_COLS:
        pub_df[c] = [0.2, 0.5, 0.8]
        spec_df[c] = [0.1, 0.6, 0.9]
    blended = blend_routed_predictions(pub_df, spec_df)
    assert len(blended) == 3
    assert not blended.isnull().any().any()
    assert list(blended.columns) == ["StudyInstanceUID"] + list(TARGET_COLS)
