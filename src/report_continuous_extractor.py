from __future__ import annotations

import re
from collections.abc import Sequence

import pandas as pd

from src.config import TARGET_COLS

INDICATION_PATTERNS = [
    r"CLINICAL\s+(?:INDICATION|HISTORY|INFORMATION).*?(?=FINDINGS|IMPRESSION|CONCLUSION|TECHNIQUE|$)",
    r"HISTORIA\s+CL[IÍ]NICA.*?(?=HALLAZGOS|CONCLUSI[OÓ]N|IMPRESI[OÓ]N|$)",
    r"INDIKATION.*?(?=BEFUND|BEURTEILUNG|ZUSAMMENFASSUNG|$)",
    r"MOTIF\s+DE\s+L'EXAMEN.*?(?=R[EÉ]SULTATS|CONCLUSION|$)",
    r"KL[Iİ]N[Iİ]K\s+B[Iİ]LG[Iİ].*?(?=BULGULAR|SONU[CÇ]|$)",
]

NEGATION_PATTERNS = [
    r"\bno\b",
    r"\bnot\b",
    r"\bwithout\b",
    r"\bfree\s+of\b",
    r"\bintact\b",
    r"\bunremarkable\b",
    r"\bnormal\b",
    r"\bpreserved\b",
    r"\bsin\b",
    r"\blibre\s+de\b",
    r"\b[ií]ntegro[s]?\b",
    r"\bkein[enrs]?\b",
    r"\bohne\b",
    r"\bunauff[aä]llig[enrs]?\b",
    r"\bregelrecht[enrs]?\b",
    r"\bsans\b",
    r"\babsence(?:\s+d['e])?\b",
    r"\bpas\s+de\b",
    r"\baucun[e]?\b",
    r"\bintacte?[s]?\b",
    r"\byok\b",
    r"\bnegatif\b",
]

HEDGING_LEVELS = {
    "definite": (
        0.95,
        [
            r"\bcomplete\b",
            r"\bfull[- ]thickness\b",
            r"\brupture[d]?\b",
            r"\bdisplaced\b",
            r"\bsevere\b",
            r"\bmarked\b",
            r"\bgross\b",
            r"\bcompleto\b",
            r"\bsevero\b",
            r"\bmarcado\b",
            r"\bkomplett\b",
            r"\bhochgradig\b",
            r"\bdeutliche?[s]?\b",
            r"\bmassive?[s]?\b",
            r"\bruptur\b",
        ],
    ),
    "suspected": (
        0.80,
        [
            r"\bprobable\b",
            r"\blikely\b",
            r"\bsuspected\b",
            r"\bmoderate\b",
            r"\bconsiderable\b",
            r"\bprobabile\b",
            r"\bmoderado\b",
            r"\bsospecha\b",
            r"\bwahrscheinlich\b",
            r"\bmittelgradig\b",
            r"\bv\.a\.\b",
            r"\bverdacht\b",
        ],
    ),
    "possible": (
        0.50,
        [
            r"\bpossible\b",
            r"\bpartial\b",
            r"\bsubtle\b",
            r"\bmild\b",
            r"\bmildly\b",
            r"\bcannot\s+exclude\b",
            r"\bcannot\s+rule\s+out\b",
            r"\bquestionable\b",
            r"\bposible\b",
            r"\bparcial\b",
            r"\bsutil\b",
            r"\bleve\b",
            r"\bm[oö]glich\b",
            r"\bfraglich\b",
            r"\bgeringgradig\b",
            r"\bdiskrete?[s]?\b",
        ],
    ),
    "trace": (
        0.15,
        [
            r"\btrace\b",
            r"\bminimal\b",
            r"\bphysiological\b",
            r"\bm[ií]nimo\b",
            r"\bfisiol[oó]gico\b",
            r"\bgering\b",
            r"\bspuren\b",
            r"\bphysiologisch\b",
        ],
    ),
}

TARGET_KEYWORDS: dict[str, list[str]] = {
    "ACL": [
        r"\b(?:anterior\s+cruciate|acl|lca|vkb|vorderes\s+kreuzband|ligamento\s+cruzado\s+anterior)\b",
    ],
    "MCL": [
        r"\b(?:medial\s+collateral|mcl|lmc|tibial\s+collateral|innenband|ligamento\s+colateral\s+medial)\b",
    ],
    "Medial Meniscus": [
        r"\b(?:medial\s+menisc\w+|menisco\s+medial|menisco\s+interno|innenmenisk\w+)\b",
    ],
    "Lateral Meniscus": [
        r"\b(?:lateral\s+menisc\w+|menisco\s+lateral|menisco\s+externo|au[sß]enmenisk\w+)\b",
    ],
    "Medial OA": [
        r"\b(?:medial\s+(?:compartment\s+)?(?:osteoarthritis|oa|arthrosis|cartilage\s+loss|joint\s+space\s+narrowing))\b",
        r"\b(?:gonartrosis\s+medial|mediale\s+gonarthrose|medialer\s+gelenkspaltverschm[aä]lerung)\b",
    ],
    "Lateral OA": [
        r"\b(?:lateral\s+(?:compartment\s+)?(?:osteoarthritis|oa|arthrosis|cartilage\s+loss|joint\s+space\s+narrowing))\b",
        r"\b(?:gonartrosis\s+lateral|laterale\s+gonarthrose|lateraler\s+gelenkspaltverschm[aä]lerung)\b",
    ],
    "PF OA": [
        r"\b(?:patellofemoral|patelofemoral|femoropatellar|retropatellar)\s+(?:osteoarthritis|oa|arthrosis|chondromalacia|cartilage\s+loss)\b",
        r"\b(?:femoropatellare?\s+arthrose|retropatellare?\s+arthrose|gonartrosis\s+patelofemoral)\b",
    ],
    "Effusion": [
        r"\b(?:joint\s+effusion|effusion|derrame\s+articular|gelenkerguss|[eé]panchement|s[iı]v[iı]|hidrartrosis)\b",
    ],
    "Synovitis": [
        r"\b(?:synovitis|sinovitis|synovial\s+thickening|engrosamiento\s+sinovial|synovialitis)\b",
    ],
    "Baker's": [
        r"\b(?:baker['s]?\s*(?:cyst)?|popliteal\s+cyst|quiste\s+de\s+baker|bakerzyste|kyste\s+de\s+baker)\b",
    ],
    "Contusion": [
        r"\b(?:bone\s+(?:marrow\s+)?(?:contusion|edema|bruise)|edema\s+[oó]seo|contusi[oó]n\s+[oó]sea|knochenkontusion|knochenmark[oö]dem)\b",
    ],
    "Fracture": [
        r"\b(?:fracture[d]?|fractura|fraktur|cortical\s+break|cortical\s+step[- ]off|trabecular\s+fracture)\b",
    ],
}


def strip_clinical_indication(text: str) -> str:
  cleaned = text
  for pattern in INDICATION_PATTERNS:
    cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE | re.DOTALL)
  return cleaned.strip()


def extract_continuous_labels(report: str) -> dict[str, float]:
  """Extract continuous calibrated probabilities in [0.0, 1.0] from clinical radiology report.

  Distinguishes:
  - Definite/Severe: 0.95
  - Suspected/Probable: 0.80
  - Possible/Subtle: 0.50
  - Trace/Minimal: 0.15
  - Negated/Normal: 0.02
  """
  if not isinstance(report, str) or not report.strip():
    return {col: 0.02 for col in TARGET_COLS}

  cleaned_report = strip_clinical_indication(report)
  sentences = re.split(r"[.;\n]+", cleaned_report)

  neg_regex = re.compile("|".join(NEGATION_PATTERNS), re.IGNORECASE)

  scores: dict[str, float] = {col: 0.02 for col in TARGET_COLS}

  for col in TARGET_COLS:
    keywords = TARGET_KEYWORDS.get(col, [])
    kw_regex = re.compile("|".join(keywords), re.IGNORECASE)

    highest_score = 0.02

    for sentence in sentences:
      s = sentence.strip()
      if not s:
        continue

      if kw_regex.search(s):
        # Pathology matched in this sentence
        if neg_regex.search(s):
          # Negation detected in same sentence
          highest_score = max(highest_score, 0.02)
        else:
          # Check trace/physiological first: e.g. 'likely physiological' or 'minimal'
          trace_regex = re.compile("|".join(HEDGING_LEVELS["trace"][1]), re.IGNORECASE)
          if trace_regex.search(s):
            score = HEDGING_LEVELS["trace"][0]
          else:
            score = 0.75  # default positive score
            for level in ["definite", "suspected", "possible"]:
              val, patterns = HEDGING_LEVELS[level]
              lvl_regex = re.compile("|".join(patterns), re.IGNORECASE)
              if lvl_regex.search(s):
                score = val
                break

          highest_score = max(highest_score, score)

    scores[col] = float(highest_score)

  return scores


def populate_continuous_dataset(
    df: pd.DataFrame,
    report_col: str = "Report",
    gold_flag_col: str = "is_gold",
    target_cols: Sequence[str] = TARGET_COLS,
) -> pd.DataFrame:
  """Populates continuous report-extracted labels while strictly preserving human gold labels."""
  res_df = df.copy()

  # Identify gold studies
  if gold_flag_col in res_df.columns:
    gold_mask = res_df[gold_flag_col].astype(bool)
  else:
    gold_mask = pd.Series(False, index=res_df.index)

  continuous_records = []
  for idx, row in res_df.iterrows():
    if gold_mask.loc[idx]:
      # Preserve existing gold targets
      record = {col: float(row[col]) for col in target_cols if col in row}
    else:
      # Extract continuous score from report
      rep = str(row[report_col]) if report_col in row else ""
      record = extract_continuous_labels(rep)
    continuous_records.append(record)

  extracted_df = pd.DataFrame(continuous_records, index=res_df.index)
  for col in target_cols:
    res_df[col] = extracted_df[col]

  return res_df
