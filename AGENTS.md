# AGENTS.md: RSNA Knee Abnormality Detection

> **The Single Source of Truth** for all AI development assistants (GitHub Copilot, Claude Code, Gemini CLI, Cursor) working in this Kaggle competition repository.

---

## 1. Executive Summary & Competition Charter

- **Competition:** RSNA Knee Abnormalities Detection (`rsna-knee-abnormality-detection`)
- **Host:** Radiological Society of North America (RSNA) in partnership with the American Society of Neuroradiology (ASNR)
- **Prize Pool:** **$77,000 USD** (Main Leaderboard + Dedicated Efficiency Track)
- **Modality:** Multi-sequence Knee Magnetic Resonance Imaging (MRI) in standard DICOM format (`StudyInstanceUID` $\to$ multiple `SeriesInstanceUID` series)
- **Dataset Scale:** **~570 GB** total raw uncompressed volume across training and hidden evaluation cohorts
- **Objective:** Predict well-calibrated continuous confidence probabilities in $[0, 1]$ for **12 clinically significant knee abnormalities**.

### The 12 Abnormality Targets
1. `ACL` (Anterior Cruciate Ligament tear / complete rupture)
2. `MCL` (Medial Collateral Ligament tear / sprain)
3. `Medial Meniscus` (Medial meniscus tear / maceration / degeneration)
4. `Lateral Meniscus` (Lateral meniscus tear / degeneration)
5. `Medial OA` (Osteoarthritis / joint space narrowing of medial compartment)
6. `Lateral OA` (Osteoarthritis / joint space narrowing of lateral compartment)
7. `PF OA` (Patellofemoral osteoarthritis / cartilage loss)
8. `Effusion` (Joint effusion / excess intra-articular fluid)
9. `Synovitis` (Synovial thickening and inflammatory hyperintensity)
10. `Baker's` (Baker's / popliteal synovial cyst)
11. `Contusion` (Subchondral bone bruise / trabecular microfracture)
12. `Fracture` (Cortical discontinuity or subchondral fracture line)

### Official Evaluation Metric
Submissions are evaluated on the unweighted **Macro-Averaged Area Under the ROC Curve (Macro ROC-AUC)** across all 12 targets:

$$\text{Final Score} = \frac{1}{12} \sum_{i=1}^{12} \text{ROC-AUC}_i$$

- **Continuous Predictions Mandate:** Submissions must contain continuous probabilities in $[0, 1]$. Binarized $0/1$ outputs crush AUC resolution.
- **Extreme Class Imbalance:** Unweighted macro-averaging implies that rare findings (e.g. `Fracture` $\approx 2.1\%$, `MCL` $\approx 4.8\%$) dictate the final leaderboard standing just as much as common findings (`Effusion` $\approx 42\%$).
- **Metric Invariant:** Always compute scores through `src.metrics.compute_competition_metric(y_true, y_pred)`. Single-class validation batches must safely return `0.5` without crashing.
- **Local Validation Gate:** Never rely on Public Leaderboard (LB) feedback alone. Every model change must show verified 5-fold cross-validation improvement logged in `docs/experiments.md`.

---

## 2. Competitive Edge & Grandmaster Architectural Decisions

Our technical approach diverges from naive public baselines through domain-informed medical imaging physics and compute engineering:

### A. 2.5D Multi-Slot Bag Architecture vs. 3D Dense Volumes
- **The 3D Pitfall (Why Dense 3D CNNs Fail on Knee MRI):** Newcomers frequently reach for 3D convolutions (MONAI UNet3D / DenseNet3D / SwinUNETR) because MRI is volumetric. Across historical RSNA medical imaging challenges (Lumbar Spine 2024, Abdominal Trauma 2023, Cervical Spine 2022), pure 3D CNNs systematically underperform 2.5D hybrids due to five fundamental bottlenecks:
  1. **Multi-Sequence Asymmetry (MRI $\neq$ CT):** CT scans are single, isotropic 3D cubes. Knee MRI is an asymmetric collection of 4–8 distinct series with varying planes and RF contrasts (Sagittal T2 FS, Coronal T1, Axial PD). Forcing them onto a shared 3D grid requires 3D spatial interpolation that blurs fine sub-millimeter structures like cortical fracture lines.
  2. **The 10:1 Anisotropy Problem:** In-plane resolution is high ($0.3\text{--}0.5\,\text{mm}$), while through-plane slice thickness is coarse ($3.0\text{--}4.5\,\text{mm}$). Isotropic $3\times 3\times 3$ 3D kernels convolve across massive $4\,\text{mm}$ anatomical gaps, degrading spatial representations.
  3. **The Pretraining Void:** 2D backbones leverage ImageNet-21k, RadImageNet (1.3M medical images), and DINOv2 self-supervised weights. 3D backbones lack massive medical pretraining; training 3D CNNs from scratch on ~4,400 studies leads to severe over-parameterization and overfitting.
  4. **VRAM Exhaustion & Batch Size 1 Instability:** Multi-series 3D volumes at $384\times 384\times 96$ consume $>30\,\text{GB}$ VRAM, forcing batch size $B=1$. Batch Normalization statistics drift, gradients become volatile, and mixed precision underperforms.
  5. **The Winning Paradigm (2.5D Spatial + 1D Anatomical Attention):** Extract high-resolution 2.5D slice triplet representations (in-plane $0.3\,\text{mm}$) and fuse inter-slice / inter-plane relationships through 1D attention pooling (`SlotHead`).
- **Our 2.5D Slot Solution:** Model each study as a bag of up to 6 canonical acquisition slots across the 3 anatomical planes. For each slot, we sample 3 consecutive slices spaced across the joint center and stack them as a 3-channel (RGB) slab.
- **The 6 Canonical Acquisition Slots:**
  1. `SAG_FLUID_FS`: Sagittal, Fluid-sensitive (T2/PD/STIR), Fat-suppressed.
  2. `COR_FLUID_FS`: Coronal, Fluid-sensitive, Fat-suppressed.
  3. `AX_FLUID_FS`: Axial, Fluid-sensitive, Fat-suppressed.
  4. `SAG_FLUID_NOFS`: Sagittal, Fluid-sensitive, Non-fat-suppressed.
  5. `COR_T1`: Coronal, T1 structural / anatomical contrast.
  6. `SAG_T1`: Sagittal, T1 structural / anatomical contrast.

### B. High-Throughput Ingestion for the 570GB Dataset
- **The Raw DICOM Bottleneck:** Parsing 570 GB of raw uncompressed DICOMs on the fly during training saturates disk I/O and throttles GPU compute.
- **Offline Caching Pipeline:**
  1. **Preprocessing Worker:** Extract canonical 2.5D 3-channel slabs once per study and cache them as compressed `.npy` or `.parquet` byte arrays (`data/processed/slabs_224/` and `slabs_384/`).
  2. **Metadata Cataloging:** Pre-index all series geometry (`ImagePositionPatient`, `ImageOrientationPatient`, `PixelSpacing`) into fast-lookup SQLite or Parquet indices.
  3. **In-Memory Streaming:** During training, `KneeMultiSlotDataset` reads pre-extracted slabs directly, keeping dual Tesla T4 / A100 GPU compute at $>95\%$ utilization.

### C. Anatomical Slot-Prior Attention Head (`SlotHead`)
- **The Pooling Dilemma:** Naive global average pooling across series dilutes diagnosis-specific signals. An ACL tear is visible primarily on Sagittal views; an MCL or meniscal tear is diagnosed on Coronal views; Patellofemoral OA is evaluated axially.
- **Slot-Attention Mechanism:**
  - Learnable diagnosis query vectors $q_o \in \mathbb{R}^H$ ($o \in \{1 \dots 12\}$) attend across extracted slot representations $h_s \in \mathbb{R}^H$.
  - **Dynamic Presence Masking:** A binary mask $m_s \in \{0, 1\}$ zeroes out attention logits for missing series, allowing softmax attention to gracefully redistribute over available slots without zero-vector degradation.
  - **Anatomical Prior Logit Bias:** Injects an inductive bias buffer `slot_prior` ($\text{logit} + 0.55$) tilting attention toward clinically established planes:
    - `ACL`: Sagittal slots (`SAG_FLUID_FS`, `SAG_FLUID_NOFS`, `SAG_T1`)
    - `MCL`: Coronal slots (`COR_FLUID_FS`, `COR_T1`)
    - `Medial / Lateral Meniscus`: Coronal & Sagittal slots
    - `PF OA`, `Effusion`, `Synovitis`: Axial & Sagittal fluid-sensitive slots
    - `Baker's Cyst`: Sagittal fluid-sensitive slots
    - `Contusion` / `Fracture`: Multi-plane fluid + structural slots

### D. Geometric Slice Ordering & Normalization
- **The Alphabetical Ordering Hazard:** DICOM files in Kaggle series folders cannot be ordered by filename or SOPInstanceUID. Doing so scrambles through-plane anatomy.
- **Geometric Normal Projection:** Slice order MUST be established by projecting `ImagePositionPatient` $(0020, 0032)$ along the normal vector computed from `ImageOrientationPatient` $(0020, 0037)$:
  $$\vec{n} = \vec{r}_{\text{row}} \times \vec{r}_{\text{col}}, \qquad \text{slice\_pos} = \vec{p}_{\text{patient}} \cdot \vec{n}$$
  Sorting slices by $\text{slice\_pos}$ guarantees consistent spatial progression.
- **Physical Millimeter Field of View (FOV):** Resizing raw pixel grids without compensating for `PixelSpacing` introduces scale variance across scanners. We apply a fixed $130\,\text{mm}$ center crop (covering the joint capsule in $>99.6\%$ of patients) before resizing to $224\times 224$ or $384\times 384$.
- **Laterality Normalization (Left-Knee Canon):** All knees are canonicalized to a left-knee orientation:
  - Right knees: Coronal and Axial planes are flipped horizontally.
  - Sagittal planes: Slice sequence order is inverted (medial-to-lateral progression runs opposite between knees).

### E. Data Augmentation Anti-Patterns
- ❌ **NEVER apply horizontal flips** after laterality normalization. A horizontal flip swaps Medial and Lateral compartments, directly corrupting `Medial Meniscus` vs. `Lateral Meniscus` and `Medial OA` vs. `Lateral OA`.
- ❌ **NEVER apply vertical flips**. Knees are acquired under standardized craniocaudal patient positioning.
- ✅ **Permitted Augmentations:** Rigid affine transformations ($\pm 8^\circ$ rotation, $\pm 5\%$ translation, $\pm 8\%$ isotropic scale) and mild contrast/brightness jitter ($\pm 10\%$).

### F. Class Imbalance & Loss Optimization
- **Asymmetric Focal Loss (ASL):** Standard Binary Cross-Entropy (BCE) gets overwhelmed by easy negative samples on rare findings (`Fracture`, `MCL`). ASL decouples focusing parameters ($\gamma_- = 4, \gamma_+ = 0$) with probability shifting, heavily penalizing false negatives on rare findings while suppressing gradient noise from easy negatives.
- **Positive Frequency Re-Weighting:** Compute inverse frequency loss weights $w_i = \sqrt{N_{\text{neg}, i} / N_{\text{pos}, i}}$ clamped to $[1.0, 8.0]$.

### G. Validation Hygiene & Leakage Prevention
- **Report Text Hash Grouping:** Training studies contain templated radiology reports where identical text describes normal studies. Splitting identical reports between train and validation leaks ground truth labels. Folds MUST be generated with `generate_report_grouped_folds()` grouped by `hashlib.md5(report.encode()).hexdigest()`.
- **Train/Test Modality Asymmetry:** Free-text radiology reports exist in training, but **test data contains NO text reports** (only raw DICOMs). NLP models (DeBERTa / clinical LLMs) can only be used offline for pseudo-labeling or distillation; the test graph strictly consumes DICOM pixels.
- **Percentile Rank Calibration:** Because ROC-AUC evaluates relative concordance rather than calibrated probabilities, always convert raw predictions to fractional percentiles (`pd.Series.rank(pct=True)`) before model blending or final submission.

### H. Multi-Domain Probit Superstack (0.954+ SOTA) & Efficiency Track
1. **CoAtNet-384 SWA (Global Anatomical Anchor):** Evaluates native $384\times 384$ DICOM volumes across 94 slice windows with Anatomical Mirror TTA, dominating whole-joint pathologies (`Effusion`, `Synovitis`, `Lateral Meniscus`).
2. **All-Data S6 ConvNeXt (Local Structural Core):** Pre-trained on clinical musculoskeletal scans with SKM-TEA anchors, providing high baseline feature extraction.
3. **2.5D ConvNeXt-Small MIL Reader (Specialized Target Lift):** Addresses CoAtNet's known blindness on `Lateral OA` (ConvNeXt achieves **0.8549 vs 0.7756**, a **+0.0793 AUC lift**) and `Baker's Cyst` (**0.984 vs 0.940**, **+0.044 AUC lift**).
4. **Latent Gaussian Probit Quantile Transformation:** Maps percentiles to latent standard normal space ($z = \Phi^{-1}(p)$ via `scipy.special.ndtri`), performs target-weighted clinical fusion, and projects back ($p = \Phi(z)$ via `ndtr`), preventing rank compression at the distribution tails.
5. **Dual-Slot Portfolio Strategy:**
   - **Slot 1 (Max Performance):** Grandmaster Probit Superstack (`Ref: 57119660`, **0.954 Macro AUC**).
   - **Slot 2 (Efficiency Track & Shakeup Hedge):** Efficiency Champion (`Ref: 57174412`, **0.946 Macro AUC**, ~7 min total turnaround, $<15\text{ ms/study}$). Ensures eligibility for the dedicated Efficiency prize.

---

## 3. Project Architecture & Technical Stack

### A. Technology Stack
- **Languages:** Python 3.10+
- **Deep Learning Framework:** PyTorch 2.1+, Torchvision
- **Model Backbones:** `timm` (ConvNeXt, CoAtNet, EfficientNet-V2, ResNet), DINOv2
- **Medical Imaging:** `pydicom`, `pylibjpeg`, `python-gdcm`, `scikit-image`
- **Volumetric Reference:** MONAI (reference benchmarking only; production models use 2.5D multi-slot)
- **Validation & Metrics:** `scikit-learn`, `iterative-stratification` (`iterstrat`)
- **Optimization:** SciPy (`scipy.optimize.minimize`, `scipy.special.ndtr`, `ndtri`)
- **Experiment Tracking:** Weights & Biases (`wandb`)

### B. Directory Layout
```text
rsna-knee-abnormality-detection/
├── .github/
│   └── copilot-instructions.md  # Symlink -> AGENTS.md
├── CLAUDE.md                    # Symlink -> AGENTS.md
├── GEMINI.md                    # Symlink -> AGENTS.md
├── AGENTS.md                    # Master instructions (Single Source of Truth)
├── pyproject.toml               # Project configuration, dependencies, ruff settings
├── requirements.txt             # Pinned package requirements
├── input/                       # Competition datasets & DICOM series (gitignored)
│   ├── train.csv                # StudyInstanceUID, Report, 12 binary labels
│   ├── train_series.csv         # StudyInstanceUID, SeriesInstanceUID, plane, flags
│   ├── train_series/            # Raw training DICOM series
│   ├── test.csv                 # Test study identifiers
│   ├── test_series.csv          # Test series descriptors
│   ├── test_series/             # Test DICOM series
│   └── sample_submission.csv    # Official submission format
├── models/                      # Trained model weights & specialist checkpoints
│   └── specialist_checkpoints/  # 5-fold specialist weights (specialist_rad_fold0..4.pth)
├── src/                         # Core production library
│   ├── __init__.py
│   ├── config.py                # Paths, constants, seeds, target column definitions
│   ├── metrics.py               # Macro ROC-AUC evaluation and target breakdown
│   ├── dataset.py               # DICOM parsing, geometric ordering, 2.5D slab extraction
│   ├── ensemble.py              # Rank blending & Probit quantile normal fusion
│   ├── optimize_routing.py      # Bounded Nelder-Mead routing weight optimizer
│   ├── report_reader.py         # Multilingual rule-based clinical NLP extractor
│   ├── landmark.py              # Joint center physical coordinate locator
│   ├── models/                  # Neural network modules
│   │   ├── __init__.py
│   │   ├── backbones.py         # Vision backbones (timm, DINOv2, ConvNeXt, RadImageNet)
│   │   ├── slot_head.py         # Slot-attention aggregator with anatomical prior
│   │   └── model.py             # End-to-end multi-slot knee classifier
│   ├── train.py                 # Multi-fold training CLI with AMP, CosineAnnealing, W&B
│   └── infer.py                 # Kaggle inference runner and submission generator
├── notebooks/                   # Submission kernels & training notebooks
│   ├── public_baselines/        # Reference community baseline notebooks
│   ├── submission/              # Verified competition submission packages
│   │   ├── grandmaster_superstack_0956/  # 0.954 Grandmaster Probit Superstack
│   │   ├── helios_0954_apex/             # 0.954 Helios Apex pipeline
│   │   ├── probit_superstack/            # 0.950 4-Reader Probit Superstack
│   │   └── apex_0950_sota/               # 0.950 Pure Apex Baseline
│   └── train_specialist/        # Complete 5-fold training kernels (Folds 0..4)
├── oof/                         # Out-of-fold validation predictions (Parquet/CSV)
├── submissions/                 # Generated competition submission files
├── docs/                        # Architectural records, intelligence, and logs
│   ├── README.md                # Master documentation index
│   ├── solution_writeup.md      # Comprehensive end-to-end technical report
│   ├── competitive_advantages_and_gap_analysis.md # Grandmaster gap analysis
│   ├── top_solutions_analysis_and_benchmarking.md # Benchmark vs top solutions
│   ├── final-submission-selection.md # Final 2-slot selection audit & efficiency hedge
│   ├── experiments.md           # Master experiment ledger (exp-01 to exp-23)
│   ├── compute_and_hardware.md  # Local M2 Pro vs Cloud hardware profile
│   └── intel/                   # Forum intel, data specs, and top notebook summaries
├── scripts/                     # Operational automation scripts
│   ├── build_grandmaster_superstack_notebook.py  # Builds 0.956+ superstack
│   ├── build_complete_specialist_notebook.py     # Parameterized 5-fold generator
│   └── setup_symlinks.sh                         # Symlink synchronization script
└── tests/                       # Pytest verification suite (62 passing unit tests)
    ├── test_metrics.py          # Metric accuracy and single-class edge cases
    ├── test_dataset.py          # Geometry ordering and slab extraction
    ├── test_models.py           # Model forward shapes, slot masking, backward pass
    ├── test_loss.py             # Asymmetric Focal Loss & inverse frequency weighting
    ├── test_ensemble.py         # Multi-model rank averaging & Probit normal fusion
    ├── test_landmark.py         # Physical joint-center locator & slab extraction
    ├── test_report_reader.py    # Multilingual report extractor rules & leakage tests
    ├── test_report_continuous_extractor.py # Continuous clinical certainty extractor
    ├── test_series_selection.py # Dynamic series and plane selection tests
    ├── test_specialist_metrics.py # Rare-target specialist validation
    └── test_optimize_routing.py # Routing optimizer validation
```

---

## 4. Code Standards & Engineering Guardrails

### A. Strict Typing & Python Conventions
- Target **Python 3.10+** with `from __future__ import annotations` at the top of every file.
- All function signatures and interfaces must carry explicit type annotations (`np.ndarray`, `torch.Tensor`, `pd.DataFrame`, `Sequence[str]`, `dict[str, float]`).
- Do not use bare `dict` or `Any` where dataclasses or typed dictionaries clarify contracts.

### B. Reproducibility & Deterministic Seeding
Every script, dataset split, and training loop must invoke deterministic seeding:
```python
import os
import random
import numpy as np
import torch


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
```

### C. DICOM Header Robustness & Error Handling
- Never assume standard DICOM tag presence across varying hospital equipment:
  - If `Laterality` is missing, infer laterality geometrically via table coordinates (`infer_laterality_from_geometry()`).
  - If a slice pixel array fails decompression (e.g. missing JPEG-LS / JPEG 2000 codecs), gracefully fallback to the nearest valid neighbor slice. **Never crash an inference batch on a single corrupted slice.**
  - Parse multi-value tags with safe fallback helpers.

### D. Formatting & Code Quality
- Lint and format using Ruff:
  ```bash
  ruff check .
  ruff format .
  ```
- Maximum line length: **100 characters**.
- No unused imports, debugging prints, or unhandled generic `except:` clauses.

### E. Test-Driven Verification
- Execute `pytest` before committing changes to ensure metric calculations, data transformations, and model forward passes remain uncompromised:
  ```bash
  pytest -v
  ```

---

## 5. Common Development Workflows

### Workflow 1: Running Cross-Validation Folds
1. Generate or verify 5-fold Multilabel Stratified Split grouped by report hash:
   ```bash
   python -m src.dataset --generate-splits --n-splits 5 --seed 42
   ```
2. Train a specific fold with AMP mixed precision and Cosine Annealing:
   ```bash
   python -m src.train \
     --model resnet34 \
     --img-size 224 \
     --fold 0 \
     --epochs 10 \
     --batch-size 8 \
     --lr 1e-4 \
     --use-amp
   ```
3. Evaluate OOF predictions against ground truth using the official metric:
   ```python
   from src.metrics import compute_detailed_metrics

   scores = compute_detailed_metrics(oof_df[TARGET_COLS], oof_df[[c + "_pred" for c in TARGET_COLS]])
   print(f"OOF Macro AUC: {scores['macro_avg']:.5f}")
   ```
4. Record model architecture, image size, and CV score in `docs/experiments.md`.

### Workflow 2: Implementing a New Backbone or Custom Head
1. Add the feature extractor backbone to `src/models/backbones.py`.
2. Ensure the output shape matches `(Batch * Slots, Channels, Spatial_H, Spatial_W)` or `(Batch, Slots, Embedding_Dim)`.
3. Feed representations through `SlotHead` in `src/models/slot_head.py`, maintaining slot presence masking:
   ```python
   logits = model(imgs=imgs_tensor, mask=slot_mask_tensor)  # (Batch, 12)
   ```
4. Run `pytest tests/test_models.py` to confirm forward and backward gradient passes.

### Workflow 3: Generating a Kaggle-Compliant Submission
1. Run local inference simulating the Kaggle test environment:
   ```bash
   python -m src.infer \
     --weights-dir models/specialist_checkpoints/ \
     --output submissions/submission.csv
   ```
2. **Mandatory Post-Inference Validation:**
   - [ ] Columns exactly match `sample_submission.csv` (`StudyInstanceUID` + 12 targets).
   - [ ] Row count exactly matches `test.csv` study count.
   - [ ] Zero `NaN`, `null`, or infinite values (impute missing predictions with `0.5`).
   - [ ] All confidence scores in $[0, 1]$.
   - [ ] Predictions calibrated via percentile ranking (`.rank(pct=True)`).

---

## 6. Symlink Configuration Setup

To ensure all AI development assistants (GitHub Copilot, Claude Code, Gemini CLI, Cursor) seamlessly reference `AGENTS.md` as the single source of truth across all operating systems and environments, execute the bundled setup script:

```bash
chmod +x scripts/setup_symlinks.sh
./scripts/setup_symlinks.sh
```

### Manual Symlink Setup Commands
If executing manually from the repository root:
```bash
# Ensure .github directory exists
mkdir -p .github

# Create relative symlinks pointing directly to AGENTS.md
ln -sf AGENTS.md CLAUDE.md
ln -sf AGENTS.md GEMINI.md
ln -sf ../AGENTS.md .github/copilot-instructions.md

# Verify configuration
ls -la CLAUDE.md GEMINI.md .github/copilot-instructions.md
```

---

## 7. AI Assistant Operating Protocol

Whenever an AI assistant operates within this repository:

1. **Consult This Document First:** Treat `AGENTS.md` as the ultimate authority on architectural patterns, constraints, and validation standards.
2. **Never Invent Metric Logic:** Always import and invoke `compute_competition_metric` and `compute_detailed_metrics` from `src.metrics`.
3. **Respect Laterality & Anatomy:** Never suggest augmentations (like horizontal flipping) that invalidate medical semantics.
4. **Efficiency Awareness:** Prioritize throughput and low memory footprints for the Efficiency Track.
5. **Always Verify Locally:** Write unit tests for new modules, verify execution, and never declare a task complete without empirical test evidence.
