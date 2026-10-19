from __future__ import annotations

import pandas as pd

from src.config import TARGET_COLS
from src.report_continuous_extractor import extract_continuous_labels, populate_continuous_dataset


def test_continuous_hedging_hierarchy():
  """Asserts that clinical hedging words map to strictly monotonic continuous probabilities."""
  rep_definite = "Complete full-thickness tear of the anterior cruciate ligament with rupture."
  rep_suspected = "Probable tear of the anterior cruciate ligament."
  rep_possible = "Possible subtle sprain of the anterior cruciate ligament, cannot exclude partial tear."
  rep_trace = (
      "Minimal signal heterogeneity of the ACL, likely physiological or prior"
      " healed sprain."
  )
  rep_negative = (
      "The anterior cruciate ligament is intact with normal thickness."
  )

  score_def = extract_continuous_labels(rep_definite)["ACL"]
  score_susp = extract_continuous_labels(rep_suspected)["ACL"]
  score_poss = extract_continuous_labels(rep_possible)["ACL"]
  score_trace = extract_continuous_labels(rep_trace)["ACL"]
  score_neg = extract_continuous_labels(rep_negative)["ACL"]

  assert score_def >= 0.90, f"Definite ACL should be >= 0.90, got {score_def}"
  assert 0.70 <= score_susp < score_def, (
      f"Suspected ACL should be [0.70, {score_def}), got {score_susp}"
  )
  assert 0.35 <= score_poss < score_susp, (
      f"Possible ACL should be [0.35, {score_susp}), got {score_poss}"
  )
  assert 0.08 <= score_trace < score_poss, (
      f"Trace ACL should be [0.08, {score_poss}), got {score_trace}"
  )
  assert score_neg <= 0.05, f"Negative ACL should be <= 0.05, got {score_neg}"


def test_multilingual_negation_continuous():
  """Verifies that German, Spanish, and French negative reports produce near-zero scores."""
  rep_spanish = "Sin evidencia de desgarro meniscal ni derrame articular. Ligamentos cruzados íntegros."
  rep_german = "Unauffällige Menisken, kein Erguss, kein Hinweis auf Fraktur oder Knochenkontusion."
  rep_french = "Intégrité des ménisques et des ligaments croisés. Absence d'épanchement intra-articulaire."

  res_es = extract_continuous_labels(rep_spanish)
  res_de = extract_continuous_labels(rep_german)
  res_fr = extract_continuous_labels(rep_french)

  for col in [
      "Medial Meniscus",
      "Lateral Meniscus",
      "Effusion",
      "Fracture",
      "Contusion",
  ]:
    assert res_es[col] <= 0.05, f"Spanish negation failed for {col}: {res_es[col]}"
    assert res_de[col] <= 0.05, f"German negation failed for {col}: {res_de[col]}"
    assert res_fr[col] <= 0.05, f"French negation failed for {col}: {res_fr[col]}"


def test_clinical_indication_stripping():
  """Verifies clinical indication text does not trigger false positive detections."""
  rep = (
      "CLINICAL INDICATION: Rule out lateral meniscus tear and fracture after"
      " ski fall.\nFINDINGS: Normal knee examination. No tear, no fracture, no"
      " effusion."
  )
  res = extract_continuous_labels(rep)
  assert res["Lateral Meniscus"] <= 0.05
  assert res["Fracture"] <= 0.05


def test_populate_continuous_preserves_gold_labels():
  """Asserts that human-annotated gold studies are never modified."""
  mock_df = pd.DataFrame({
      "StudyInstanceUID": ["gold_01", "normal_report_02"],
      "Report": [
          "Completely normal study without abnormality.",
          "Full thickness tear of the medial meniscus.",
      ],
      "ACL": [1.0, 0.0],
      "Medial Meniscus": [0.0, 0.0],
      "is_gold": [True, False],
  })
  for col in TARGET_COLS:
    if col not in mock_df.columns:
      mock_df[col] = 0.0

  populated = populate_continuous_dataset(mock_df)

  # gold_01 has gold ACL=1.0 and Medial Meniscus=0.0 despite normal report
  assert populated.loc[0, "ACL"] == 1.0
  assert populated.loc[0, "Medial Meniscus"] == 0.0

  # normal_report_02 is non-gold, so its Medial Meniscus must be continuous from report (>0.90)
  assert populated.loc[1, "Medial Meniscus"] >= 0.90
