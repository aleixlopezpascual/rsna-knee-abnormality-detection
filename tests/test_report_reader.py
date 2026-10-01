from __future__ import annotations

import pandas as pd

from src.config import TARGET_COLS
from src.report_reader import extract_from_report, populate_labels_from_reports


def test_extract_from_report_english_acl_tear():
    report = """
    FINDINGS:
    Complete tear of the anterior cruciate ligament with surrounding joint effusion.
    The medial and lateral menisci are intact with no tears.
    No fracture or dislocation seen.
    """
    scores = extract_from_report(report)
    assert scores["ACL"] > 0.8
    assert scores["Effusion"] > 0.5
    assert scores["Medial Meniscus"] < 0.25
    assert scores["Lateral Meniscus"] < 0.25
    assert scores["Fracture"] < 0.25


def test_extract_from_report_spanish_meniscus():
    report = """
    Rotura compleja del menisco medial con derrame articular moderado.
    Ligamentos cruzados conservados. Sin fractura.
    """
    scores = extract_from_report(report)
    assert scores["Medial Meniscus"] > 0.7
    assert scores["Effusion"] > 0.6
    assert scores["ACL"] < 0.25
    assert scores["Fracture"] < 0.25


def test_extract_from_report_german_oa():
    report = """
    Ausgeprägte Gonarthrose des medialen Kompartiments mit Knorpelverlust.
    Bakerzyste in der Kniekehle nachweisbar.
    """
    scores = extract_from_report(report)
    assert scores["Medial OA"] > 0.7
    assert scores["Baker's"] > 0.7


def test_populate_labels_preserves_gold():
    df = pd.DataFrame({
        "StudyInstanceUID": ["study_01", "study_02"],
        "Report": [
            "Complete tear of the ACL.",
            "Normal knee MRI, intact ligaments.",
        ],
        "ACL": [1.0, float("nan")],
    })
    populated = populate_labels_from_reports(df, target_cols=TARGET_COLS)
    # Human gold 1.0 must be preserved
    assert populated.loc[0, "ACL"] == 1.0
    # Study 2 ACL was NaN, should now be extracted as low score
    assert populated.loc[1, "ACL"] < 0.3
    # Check that all target cols exist
    for col in TARGET_COLS:
        assert col in populated.columns
        assert f"{col}__conf" in populated.columns
