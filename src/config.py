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
