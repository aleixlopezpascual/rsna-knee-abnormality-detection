# RSNA Knee Abnormalities Detection AI Challenge

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-orange.svg)](https://pytorch.org/)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![Tests: 37 passed](https://img.shields.io/badge/tests-37%20passed-green.svg)](https://docs.pytest.org/)
[![Public LB: 0.954](https://img.shields.io/badge/Public%20LB-0.954-gold.svg)](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection)
[![Kaggle: $77,000 USD](https://img.shields.io/badge/Kaggle-RSNA%20Knee%20Detection-blue)](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection)

High-performance machine learning repository for the **Radiological Society of North America (RSNA) Knee Abnormalities Detection** competition. This codebase delivers a production-grade, modular framework for volumetric MRI multi-label classification across 12 clinically significant knee pathologies, verified at **0.954 Macro ROC-AUC** on the official Kaggle Public Leaderboard.

---

## 🏆 Official Leaderboard Progression & Results

| Ref | Submission / Experiment | Public Score | Status | Key Innovation |
| :---: | :--- | :---: | :---: | :--- |
| **`57119660`** | **`exp-13 Grandmaster Probit Superstack`** | **0.954** 🏆 | **COMPLETE** | **4-Pillar Multi-Domain Fusion:** Fuses 0.954 Helios Baseline (80–98%) + 2.5D ConvNeXt Reader (+0.079 Lateral OA) + complete 5-Fold ASL Specialist (Fracture +0.032) in Latent Gaussian Probit space |
| **`57114576`** | **`exp-12 Helios 0.954 Apex`** | **0.954** 🏆 | **COMPLETE** | SOTA CoAtNet-384 SWA (0.70) + OrthoFoundation S6 ConvNeXt (0.30) |
| `57078027` | `exp-11 4-Reader Probit Superstack` | **0.950** | `COMPLETE` | CoAtNet-384 + CoAtNet-224 + ConvNeXt + 5-Fold Specialist |
| `57047082` | `exp-10 Pure Apex SOTA Baseline` | **0.950** | `COMPLETE` | Established verified 0.950 SOTA anchor |
| `57019581` | `exp-09 Apex Specialist (2-Fold)` | **0.948** | `COMPLETE` | First leap above 0.943 ceiling |
| `56907494` | `exp-06 Golden Fusion v3` | **0.943** | `COMPLETE` | Foundation consensus (DINOv2 + Raptor + CoAtNet) |
| `56748754` | `exp-01 Baseline v1` | **0.891** | `COMPLETE` | Initial 2.5D baseline |

---

## 📖 Table of Contents
- [1. Competition Overview](#1-competition-overview)
- [2. The 12 Abnormality Targets](#2-the-12-abnormality-targets)
- [3. Key Architectural Innovations](#3-key-architectural-innovations)
- [4. Project Layout](#4-project-layout)
- [5. Getting Started](#5-getting-started)
- [6. Development & Training Workflows](#6-development--training-workflows)
- [7. Kaggle Kernel Submission Workflow](#7-kaggle-kernel-submission-workflow)
- [8. Testing & Code Quality](#8-testing--code-quality)

---

## 1. Competition Overview

- **Challenge:** RSNA Knee Abnormalities Detection (`rsna-knee-abnormality-detection`)
- **Host:** Radiological Society of North America (RSNA)
- **Prize Pool:** **$77,000 USD** (Main Track + Efficiency Track)
- **Objective:** Predict per-study confidence probabilities in $[0, 1]$ across 12 distinct abnormality targets from multi-sequence knee MRI studies (`StudyInstanceUID`).
- **Official Evaluation Metric:** Macro-Averaged ROC-AUC:
  $$\text{Macro ROC-AUC} = \frac{1}{12} \sum_{i=1}^{12} \text{ROC-AUC}_i$$
- **Train/Test Modality Asymmetry:**
  - **Training Split:** Includes both volumetric DICOM sequences and original free-text radiology reports across 9 languages. Only 58 studies have human gold annotations, requiring NLP weak supervision or distillation.
  - **Test Split:** **Strictly image-only** (no text reports). All test-time inference must operate solely on raw MRI DICOM series across Sagittal, Coronal, and Axial planes.

---

## 2. The 12 Abnormality Targets

| Category | Target Name | Description | Key Anatomical Plane(s) |
| :--- | :--- | :--- | :--- |
| **Ligaments** | `ACL` | Anterior Cruciate Ligament tear/injury | Sagittal |
| | `MCL` | Medial Collateral Ligament tear/injury | Coronal |
| **Menisci** | `Medial Meniscus` | Medial meniscus tear or degeneration | Coronal & Sagittal |
| | `Lateral Meniscus` | Lateral meniscus tear or degeneration | Coronal & Sagittal |
| **Osteoarthritis** | `Medial OA` | Osteoarthritis of medial tibiofemoral joint | Coronal |
| | `Lateral OA` | Osteoarthritis of lateral tibiofemoral joint | Coronal |
| | `PF OA` | Patellofemoral compartment osteoarthritis | Axial & Sagittal |
| **Joint & Bone** | `Effusion` | Excess intra-articular synovial fluid | Axial & Sagittal |
| | `Synovitis` | Synovial membrane inflammation | Axial & Sagittal |
| | `Baker's` | Popliteal / Baker's cyst in knee joint | Sagittal |
| | `Contusion` | Trabecular bone bruise / marrow edema | Multi-plane fluid |
| | `Fracture` | Cortical or subchondral bone fracture | Multi-plane fluid & T1 |

---

## 3. Key Architectural Innovations

### A. 2.5D Multi-Slot Bag Architecture (`SlotHead`)
Rather than processing dense 3D volumes (which exhaust GPU VRAM and bottle on the ~247GB dataset), we formulate each study as a bag of up to 6 canonical acquisition slots:
1. `SAG_FLUID_FS`: Sagittal Fluid-Sensitive, Fat-Suppressed
2. `COR_FLUID_FS`: Coronal Fluid-Sensitive, Fat-Suppressed
3. `AX_FLUID_FS`: Axial Fluid-Sensitive, Fat-Suppressed
4. `SAG_FLUID_NOFS`: Sagittal Fluid-Sensitive, Non-Fat-Suppressed
5. `COR_T1`: Coronal T1 structural
6. `SAG_T1`: Sagittal T1 structural

### B. Anatomical Prior Attention Head
- 12 learnable diagnosis query tokens $q_o \in \mathbb{R}^H$ attend across slot representations $h_s \in \mathbb{R}^H$.
- **Dynamic Masking:** Softmax attention automatically zeroes out absent series without degraded dummy tokens.
- **Clinical Prior Logit Bias:** Injects an inductive anatomical bias (`SLOT_PRIOR_TABLE`) mapping clinical expertise directly into query-key attention matrices.

### C. True Geometric Slice Ordering
Avoids the fatal alphabetical/SOPInstanceUID sorting bug. Computes slice positions by projecting `ImagePositionPatient` $(0020, 0032)$ along the cross-product normal $\vec{n} = \vec{r}_{\text{row}} \times \vec{r}_{\text{col}}$ from `ImageOrientationPatient` $(0020, 0037)$:
$$\text{slice\_pos} = \vec{p}_{\text{patient}} \cdot \vec{n}$$

### D. FOV Physical Scale & Laterality Normalization
- **Physical Scale:** Applies a fixed $130\,\text{mm}$ center crop based on `PixelSpacing` to ensure lesion dimensions are physically consistent across scanner vendors.
- **Left-Knee Canon:** Standardizes all right knees to match left knees (horizontally flips Coronal/Axial views; reverses Sagittal slice order).
- **Prohibited Augmentations:** Strictly bans horizontal and vertical flips during training, which would invalidate medial vs. lateral distinctions.

### E. Leak-Free Validation Grouping
Groups identical boilerplate normal radiology reports by MD5 hash (`MD5(Report)`), ensuring template reports never cross train/validation splits while applying `MultilabelStratifiedKFold`.

---

## 4. Project Layout

```
rsna-knee-abnormality-detection/
├── .github/
│   └── copilot-instructions.md  # Symlink to AGENTS.md
├── AGENTS.md                    # Single Source of Truth for all AI assistants
├── CLAUDE.md                    # Symlink to AGENTS.md
├── GEMINI.md                    # Symlink to AGENTS.md
├── README.md                    # Project documentation & benchmark overview
├── pyproject.toml               # Project metadata, pytest & ruff configuration
├── requirements.txt             # Locked Python virtual environment dependencies
├── scripts/
│   └── setup_symlinks.sh        # Automates AI assistant instructions symlinking
├── input/                       # Competition datasets (DICOM series & CSVs)
│   ├── train.csv                # 4,407 training studies + reports + gold labels
│   ├── train_series.csv         # Series metadata & acquisition flags
│   ├── test.csv                 # Test study identifiers
│   ├── test_series.csv          # Test series descriptors
│   └── sample_submission.csv    # Official submission format
├── src/                         # Core modular library
│   ├── config.py                # Paths, constants, seeds, target column definitions
│   ├── metrics.py               # Competition ROC-AUC evaluation and scoring
│   ├── report_reader.py         # Multilingual rule-based clinical NLP extractor
│   ├── dataset.py               # DICOM geometry, laterality canon, slot datasets
│   ├── models/                  # Neural network modules
│   │   ├── backbones.py         # 2D vision backbones (timm, DINOv2, ConvNeXt, ResNet)
│   │   ├── slot_head.py         # Per-diagnosis SlotHead attention with priors
│   │   └── model.py             # End-to-end multi-slot knee classifier
│   ├── train.py                 # Multi-fold training CLI with AMP, CosineAnnealing, OOF
│   └── infer.py                 # Kaggle inference runner & submission validator
├── notebooks/                   # Public baselines and submission packages
│   ├── public_baselines/        # Reference community baseline notebooks
│   └── submission/              # Active Kaggle submission package & metadata
├── oof/                         # Out-of-fold validation predictions (.parquet)
├── submissions/                 # Generated competition submission files (.csv)
├── docs/                        # Architectural records, intelligence, and logs
│   ├── competition-brief.md     # High-level overview and specifications
│   ├── experiments.md           # Master experiment ledger (CV scores & parameters)
│   └── intel/                   # Forum intel, data specs, and top notebook summaries
└── tests/                       # Pytest verification suite (18 passing tests)
    ├── test_metrics.py          # Metric accuracy and single-class edge cases
    ├── test_dataset.py          # Geometry ordering, laterality flip, 5-fold split
    ├── test_models.py           # Model forward shapes, slot masking, backward pass
    └── test_report_reader.py    # Report NLP extraction across multiple languages
```

---

## 5. Getting Started

### Prerequisites
- Python 3.10+
- [uv](https://github.com/astral-sh/uv) (recommended) or standard `python3 -m venv`
- Kaggle CLI configured (`~/.kaggle/kaggle.json`)

### Installation
```bash
# Clone the repository
git clone https://github.com/aleixlopez/rsna-knee-abnormality-detection.git
cd rsna-knee-abnormality-detection

# Create virtual environment and install dependencies
uv venv .venv
source .venv/bin/activate
uv pip install -r requirements.txt

# Configure AI assistant instructions symlinks
chmod +x scripts/setup_symlinks.sh
./scripts/setup_symlinks.sh
```

---

## 6. Development & Training Workflows

### 1. Download Competition Metadata
```bash
kaggle competitions download -c rsna-knee-abnormality-detection -f train.csv -p input/
kaggle competitions download -c rsna-knee-abnormality-detection -f train_series.csv -p input/
kaggle competitions download -c rsna-knee-abnormality-detection -f test.csv -p input/
kaggle competitions download -c rsna-knee-abnormality-detection -f test_series.csv -p input/
kaggle competitions download -c rsna-knee-abnormality-detection -f sample_submission.csv -p input/
```

### 2. Run a Dry-Run Smoke Test
Test the full training and evaluation loop on synthetic studies in seconds:
```bash
python -m src.train --model resnet18 --img-size 64 --epochs 1 --batch-size 4 --dry-run
```

### 3. Train a Real Cross-Validation Fold
Train Fold 0 with mixed precision and differential backbone learning rates:
```bash
python -m src.train \
    --model resnet34 \
    --img-size 224 \
    --fold 0 \
    --epochs 10 \
    --batch-size 8 \
    --lr 1e-4 \
    --lr-backbone 1e-5 \
    --amp
```

### 4. Local Inference & Submission Generation
Generate a validated submission file from trained weights:
```bash
python -m src.infer \
    --model resnet34 \
    --weights-path models/resnet34_fold0_best.pth \
    --output submissions/submission.csv \
    --rank-pct
```

---

## 7. Kaggle Kernel Submission Workflow

Because RSNA Knee Abnormalities Detection is a **Kaggle Code Competition**, submissions are evaluated by Kaggle's backend re-running an inference notebook against the hidden test dataset:

### 1. Push Submission Kernel to Kaggle
```bash
kaggle kernels push -p notebooks/submission/
```

### 2. Submit the Output to the Competition
Once the kernel finishes running:
```bash
kaggle competitions submit \
    -c rsna-knee-abnormality-detection \
    -k aleixlopez/rsna-knee-baseline-submission \
    -f submission.csv \
    -v 1 \
    -m "exp-01 baseline v1 ensemble"
```

### 3. Check Live Leaderboard Status
```bash
kaggle competitions submissions -c rsna-knee-abnormality-detection
```

---

## 8. Testing & Code Quality

All changes must pass strict unit testing and linting before merging or submitting:

```bash
# Run the complete test suite (37 unit tests)
pytest -v

# Run linting and formatting checks
ruff check src/ tests/ scripts/
```

---

## 📄 License
This repository is licensed under the [CC-BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/) license in accordance with official competition rules.
