from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INPUT_DIR = ROOT / "input"
SRC_DIR = ROOT / "src"
DOCS_DIR = ROOT / "docs"
OOF_DIR = ROOT / "oof"
SUBMISSION_DIR = ROOT / "submissions"
MODELS_DIR = ROOT / "models"

SEED = 42
ID_COL = "StudyInstanceUID"
TARGET_COLS = [
    "ACL",
    "MCL",
    "Medial Meniscus",
    "Lateral Meniscus",
    "Medial OA",
    "Lateral OA",
    "PF OA",
    "Effusion",
    "Synovitis",
    "Baker's",
    "Contusion",
    "Fracture",
]

# SOTA clinical routing weights (CoAtNet/CNN family weight vs. Transformer family)
COATNET_ROUTING_WEIGHTS: dict[str, float] = {
    "__default__": 0.60,
    "ACL": 0.75,
    "MCL": 0.75,
    "Medial Meniscus": 0.80,
    "Lateral Meniscus": 1.00,
    "Medial OA": 0.72,
    "Lateral OA": 0.75,
    "Fracture": 0.78,
    "Contusion": 0.65,
}


def get_target_routing_weights() -> dict[str, float]:
    """Get per-target ensembling weights for CNN/CoAtNet vs Transformer models."""
    default_w = COATNET_ROUTING_WEIGHTS["__default__"]
    return {col: COATNET_ROUTING_WEIGHTS.get(col, default_w) for col in TARGET_COLS}

