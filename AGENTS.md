# AGENTS.md: RSNA Knee Abnormality Detection

> **The Single Source of Truth** for all AI development assistants (GitHub Copilot, Claude Code, Gemini CLI, Cursor) working in this Kaggle competition repository.

---

## 1. Executive Summary & Competition Charter

- **Competition:** RSNA Knee Abnormalities Detection (`rsna-knee-abnormality-detection`)
- **Host:** Radiological Society of North America (RSNA)
- **Prize Pool:** $77,000 USD (Main Leaderboard + Efficiency Track)
- **Deadline:** October 2026
- **Objective:** Predict confidence probabilities in $[0, 1]$ for **12 clinically significant knee abnormalities** from multi-sequence MRI studies (`StudyInstanceUID`).

### The 12 Abnormality Targets
1. `ACL` (Anterior Cruciate Ligament injury/tear)
2. `MCL` (Medial Collateral Ligament injury/tear)
3. `Medial Meniscus` (Medial meniscus tear/degeneration)
4. `Lateral Meniscus` (Lateral meniscus tear/degeneration)
5. `Medial OA` (Osteoarthritis of the medial compartment)
6. `Lateral OA` (Osteoarthritis of the lateral compartment)
7. `PF OA` (Patellofemoral osteoarthritis)
8. `Effusion` (Joint effusion / excess intra-articular fluid)
9. `Synovitis` (Inflammation of the synovial lining)
10. `Baker's` (Baker's / popliteal cyst)
11. `Contusion` (Bone bruise / trabecular microfracture)
12. `Fracture` (Cortical / subchondral bone fracture)

### Official Evaluation Metric
Submissions are evaluated on **Macro-Averaged ROC-AUC** across all 12 targets:
$$\text{Final Score} = \frac{1}{12} \sum_{i=1}^{12} \text{ROC-AUC}_i$$

- **Continuous Predictions:** Model outputs must be continuous probabilities in $[0, 1]$, never thresholded binary $0/1$ labels.
- **Class Imbalance Sensitivity:** Unweighted macro-average means rare abnormalities (e.g., `Fracture`, `MCL`) affect the final score just as heavily as frequent abnormalities (`Effusion`).
- **Metric Implementation:** Always use `src.metrics.compute_competition_metric(y_true, y_pred)`. Degenerate single-class batches must safely return `0.5` without throwing errors.
- **The Golden Rule:** Never trust public LB fluctuations without local CV correlation. Every experiment and submission must be verified with 5-fold cross-validation and recorded in `docs/experiments.md`.

---

## 2. Competitive Edge & Grandmaster Technical Decisions

Our repository diverges from naive public baselines through domain-informed medical imaging decisions:

### A. 2.5D Multi-Slot Bag Architecture vs. 3D Dense Volumes
- **The 3D Pitfall:** Training full 3D CNNs or dense volumetric transformers (e.g. MONAI UNet/DenseNet) across a ~247GB–570GB dataset creates severe I/O bottlenecks, massive VRAM consumption, and limits batch sizes to $1\text{--}2$ studies, leading to noisy and unstable optimization.
- **Our 2.5D Slot Solution:** Treat each study as a bag of up to 6 canonical acquisition slots across the 3 anatomical planes. For each slot, we sample 3 consecutive slices (spaced across the anatomical region of interest) and stack them as a 3-channel (RGB) slab.
- **The 6 Canonical Slots:**
  1. `SAG_FLUID_FS`: Sagittal, Fluid-sensitive (T2/PD/STIR), Fat-suppressed.
  2. `COR_FLUID_FS`: Coronal, Fluid-sensitive, Fat-suppressed.
  3. `AX_FLUID_FS`: Axial, Fluid-sensitive, Fat-suppressed.
  4. `SAG_FLUID_NOFS`: Sagittal, Fluid-sensitive, Non-fat-suppressed.
  5. `COR_T1`: Coronal, T1 structural / anatomical contrast.
  6. `SAG_T1`: Sagittal, T1 structural / anatomical contrast.

### B. Anatomical Slot-Prior Attention Head (`SlotHead`)
- **The Pooling Pitfall:** Naive global average pooling or flat concatenation across series dilutes local pathology. An ACL tear is diagnosed almost exclusively on Sagittal views; an MCL or meniscal tear is primary on Coronal views; Patellofemoral OA is inspected axially.
- **Our Slot-Attention Mechanism:**
  - Diagnosis-specific query token vectors $q_o \in \mathbb{R}^H$ ($o \in \{1 \dots 12\}$) attend across the extracted slot representations $h_s \in \mathbb{R}^H$.
  - **Dynamic Presence Masking:** A binary mask $m_s \in \{0, 1\}$ zeroes out logits for absent sequences, allowing the softmax to gracefully re-distribute attention across existing sequences without zero-vector degradation.
  - **Clinical Anatomical Prior Bias:** We inject an inductive bias buffer `slot_prior` ($\text{logit} + 0.55$) tilting the attention logits toward established clinical planes:
    - `ACL`: Sagittal slots (`SAG_FLUID_FS`, `SAG_FLUID_NOFS`, `SAG_T1`)
    - `MCL`: Coronal slots (`COR_FLUID_FS`, `COR_T1`)
    - `Medial / Lateral Meniscus`: Coronal & Sagittal slots
    - `PF OA`, `Effusion`, `Synovitis`: Axial & Sagittal fluid-sensitive slots
    - `Baker's Cyst`: Sagittal fluid slots
    - `Contusion` / `Fracture`: Multi-plane fluid + structural slots

### C. Rigorous DICOM Geometric Slice Ordering
- **The Alphabetical Ordering Bug:** DICOM files in Kaggle series folders cannot be sorted by filename or SOPInstanceUID. Doing so scrambles slice anatomy.
- **The Geometric Solution:** Slice ordering MUST be determined by projecting `ImagePositionPatient` $(0020, 0032)$ along the normal vector computed from `ImageOrientationPatient` $(0020, 0037)$:
  $$\vec{n} = \vec{r}_{\text{row}} \times \vec{r}_{\text{col}}, \qquad \text{slice\_pos} = \vec{p}_{\text{patient}} \cdot \vec{n}$$
  Sorting slices by `slice_pos` guarantees a anatomically consistent continuous 3D traversal.

### D. FOV Physical Scale & Laterality Normalization
1. **Physical Millimeter Field of View:** Standard pixel resizing without accounting for `PixelSpacing` causes variable physical zoom levels across scanner models. We apply a fixed $130\,\text{mm}$ center crop (which covers the knee joint in $>99.6\%$ of studies) before resizing to $224 \times 224$ or $336 \times 336$.
2. **Laterality Normalization (Left-Knee Canon):** All knees are canonicalized to a left-knee orientation:
   - For Right knees: Coronal and Axial planes are flipped horizontally.
   - For Sagittal planes: The slice sequence order is reversed (medial-to-lateral progression runs opposite between knees).

### E. Data Augmentation: Crucial Anti-Patterns
- ❌ **NEVER apply horizontal flips** during training after laterality normalization. A horizontal flip swaps Medial and Lateral anatomy, destroying the distinction between `Medial Meniscus` vs. `Lateral Meniscus` and `Medial OA` vs. `Lateral OA`!
- ❌ **NEVER apply vertical flips**. Knees are imaged under standard craniocaudal patient positioning.
- ✅ **Permitted Augmentations:** Rigid affine jitter ($\pm 8^\circ$ rotation, $\pm 5\%$ translation, $\pm 8\%$ isotropic scale) and intensity jitter ($\pm 10\%$).

### F. Validation Discipline & Leakage Guards
- **Shared Report Grouping:** Training data contains radiology reports where identical standard templated texts describe normal knees. Splitting identical reports between train and validation leaks ground-truth text targets. Cross-validation splits MUST be grouped by report text hash (`hashlib.md5(report.encode()).hexdigest()`) or use Multilabel Stratified Split on study labels.
- **Train/Test Modality Asymmetry:** Free-text radiology reports are provided in training, but **test data contains NO text reports** (only raw DICOMs). Text encoders (e.g. DeBERTa/LLM readers) may only be used as offline pseudo-labelers or distillation teachers; the inference graph must strictly accept MRI DICOMs.
- **Percentile Rank Calibration:** ROC-AUC measures relative rank ordering. Convert raw prediction probabilities to fractional percentiles (`pd.Series.rank(pct=True)`) before ensembling distinct models or generating final submissions.

---

## 3. Project Architecture & Directory Layout

```
rsna-knee-abnormality-detection/
├── .github/
│   └── copilot-instructions.md  # Symlink -> AGENTS.md
├── CLAUDE.md                    # Symlink -> AGENTS.md
├── GEMINI.md                    # Symlink -> AGENTS.md
├── AGENTS.md                    # Master instructions (Single Source of Truth)
├── pyproject.toml               # Project configuration, dependencies, ruff settings
├── input/                       # Competition datasets & DICOM series
│   ├── train.csv                # StudyInstanceUID, Report, 12 binary labels
│   ├── train_series.csv         # StudyInstanceUID, SeriesInstanceUID, plane, flags
│   ├── train_series/            # Raw training DICOM series
│   ├── test.csv                 # Test study identifiers
│   ├── test_series.csv          # Test series descriptors
│   ├── test_series/             # Test DICOM series
│   └── sample_submission.csv    # Official submission format
├── src/                         # Core reusable library
│   ├── __init__.py
│   ├── config.py                # Paths, constants, seeds, target column definitions
│   ├── metrics.py               # Competition ROC-AUC evaluation and scoring
│   ├── dataset.py               # DICOM parsing, geometric ordering, 2.5D slab caching
│   ├── models/                  # Neural network modules
│   │   ├── __init__.py
│   │   ├── backbones.py         # 2D/2.5D vision backbones (timm, DINOv2, ConvNeXt)
│   │   ├── slot_head.py         # Slot-attention aggregator with anatomical prior
│   │   └── model.py             # End-to-end multi-slot knee classifier
│   ├── train.py                 # Multi-fold training CLI with AMP, CosineAnnealing, W&B
│   └── infer.py                 # Kaggle inference runner and submission generator
├── notebooks/                   # Exploratory and public reference kernels
│   └── public_baselines/        # Harvested high-scoring public notebooks
├── oof/                         # Out-of-fold validation predictions (Parquet/CSV)
├── submissions/                 # Generated competition submission files
├── docs/                        # Architectural records, intelligence, and logs
│   ├── competition-brief.md     # High-level overview and specifications
│   ├── experiments.md           # Master experiment ledger (CV scores & parameters)
│   └── intel/                   # Forum intel, data specs, and top notebook summaries
├── scripts/                     # Operational automation scripts
│   └── setup_symlinks.sh        # Symlink synchronization script
└── tests/                       # Pytest verification suite
    ├── test_metrics.py          # Metric accuracy and single-class edge cases
    ├── test_dataset.py          # Geometry ordering and slab extraction
    └── test_models.py           # Model forward shapes, slot masking, backward pass
```

---

## 4. Code Standards & Engineering Guardrails

### A. Strict Typing & Python Conventions
- Target **Python 3.10+**.
- Always include `from __future__ import annotations`.
- All function definitions, class methods, and interfaces must carry explicit type annotations (`np.ndarray`, `torch.Tensor`, `pd.DataFrame`, `Sequence[str]`, `dict[str, float]`).
- Avoid `Any` or untyped dictionary bags when domain classes or `dataclass`/`NamedTuple` provide clear structural contracts.

### B. Reproducibility & Determinism
Every experiment, dataset split, and training script must invoke deterministic seeding:
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
- Never assume standard DICOM tag presence. The dataset originates from multiple hospital centers worldwide:
  - If `Laterality` is missing, infer laterality geometrically via `ImagePositionPatient` relative to table coordinates.
  - If slice pixel arrays fail to decompress (e.g., unsupported JPEG Lossless / JPEG 2000 codecs), catch exceptions cleanly and fall back to the nearest decoded neighbor slice. Never crash an inference job on a single corrupt slice!
  - Parse multi-value tags using robust fallback helpers.

### D. Formatting & Code Quality
- Lint and format using Ruff:
  ```bash
  ruff check .
  ruff format .
  ```
- Maximum line length: **100 characters**.
- No unused imports, debugging prints, or unhandled generic `except:` clauses.

### E. Test-Driven Verification
- Always execute `pytest` before committing changes to ensure metric calculation, data transformations, and model forward passes remain uncompromised:
  ```bash
  pytest -v
  ```

---

## 5. Common Development Workflows

### Workflow 1: Running Cross-Validation Folds
1. Validate that the dataset splits exist or generate a 5-fold Multilabel Stratified Split grouped by report hash:
   ```bash
   python -m src.dataset --generate-splits --n-splits 5 --seed 42
   ```
2. Train a specific fold (or all folds) using AMP mixed precision:
   ```bash
   python -m src.train --model resnet34 --img-size 224 --fold 0 --epochs 10 --batch-size 8 --lr 1e-4
   ```
3. Evaluate OOF predictions against the ground truth using official metrics:
   ```python
   from src.metrics import compute_detailed_metrics

   scores = compute_detailed_metrics(oof_df[TARGET_COLS], oof_df[[c + "_pred" for c in TARGET_COLS]])
   print(f"OOF Macro AUC: {scores['macro_avg']:.5f}")
   ```
4. Record model architecture, resolution, and local CV score in `docs/experiments.md`.

### Workflow 2: Implementing a New Backbone or Custom Head
1. Add the feature extractor backbone to `src/models/backbones.py`.
2. Ensure the output shape matches `(Batch * Slots, Channels, Spatial_H, Spatial_W)` or `(Batch, Slots, Embedding_Dim)`.
3. Feed the features through `SlotHead` in `src/models/slot_head.py`, preserving slot presence masking:
   ```python
   logits = model(imgs=imgs_tensor, mask=slot_mask_tensor)  # (Batch, 12)
   ```
4. Run `pytest tests/test_models.py` to confirm forward and backward gradient passes.

### Workflow 3: Generating a Kaggle-Compliant Submission
1. Run local inference simulating the Kaggle test environment:
   ```bash
   python -m src.infer --weights-dir models/resnet34_5fold/ --output submissions/submission_resnet34.csv
   ```
2. **Mandatory Post-Inference Validation:**
   - [ ] Exactly matches `sample_submission.csv` columns (`StudyInstanceUID` + 12 targets).
   - [ ] Row count equals `test.csv` study count.
   - [ ] No `NaN`, `null`, or infinite values (impute missing predictions with `0.5`).
   - [ ] All confidence scores in $[0, 1]$.
   - [ ] Column predictions calibrated via percentile ranking (`.rank(pct=True)`).

---

## 6. AI Assistant Operating Protocol

Whenever an AI assistant (Copilot, Claude, Gemini, Cursor) operates within this repository:

1. **Consult This Document First:** Treat `AGENTS.md` as the ultimate authority on architectural patterns, constraints, and validation standards.
2. **Never Invent Metric Logic:** Always import and invoke `compute_competition_metric` and `compute_detailed_metrics` from `src.metrics`.
3. **Respect Laterality & Anatomy:** Never suggest augmentations (like horizontal flipping) that invalidate medical semantics.
4. **Efficiency Awareness:** Keep models lean for the Efficiency Track; prioritize high throughput and low memory footprints.
5. **Always Verify Locally:** Write unit tests for new modules, verify execution, and never declare a task complete without empirical test evidence.
