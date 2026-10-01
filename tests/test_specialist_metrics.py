from __future__ import annotations

from scripts.evaluate_specialist_oof import evaluate_specialist_gate


def test_specialist_gate_criteria_pass():
    baseline_scores = {
        "Fracture": 0.840,
        "MCL": 0.875,
        "Contusion": 0.890,
        "Lateral Meniscus": 0.905,
        "macro_avg": 0.920,
    }
    specialist_scores = {
        "Fracture": 0.872,
        "MCL": 0.901,
        "Contusion": 0.905,
        "Lateral Meniscus": 0.915,
        "macro_avg": 0.935,
    }
    passed, deltas = evaluate_specialist_gate(
        baseline_scores, specialist_scores, min_rare_delta=0.02
    )
    assert passed is True
    assert deltas["Fracture"] >= 0.02
    assert deltas["MCL"] >= 0.02


def test_specialist_gate_criteria_fail():
    baseline_scores = {
        "Fracture": 0.840,
        "MCL": 0.875,
        "Contusion": 0.890,
        "Lateral Meniscus": 0.905,
        "macro_avg": 0.920,
    }
    specialist_scores = {
        "Fracture": 0.845,
        "MCL": 0.876,
        "Contusion": 0.890,
        "Lateral Meniscus": 0.905,
        "macro_avg": 0.921,
    }
    passed, _deltas = evaluate_specialist_gate(
        baseline_scores, specialist_scores, min_rare_delta=0.02
    )
    assert passed is False
