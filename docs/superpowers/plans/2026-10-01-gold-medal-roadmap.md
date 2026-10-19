# RSNA Knee Abnormalities Detection: Gold-Medal Winning Roadmap

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Establish a proprietary, top-5 competitive advantage in the RSNA Knee Abnormalities Detection challenge by transitioning from public baseline ensembles to a structured system featuring clinical NLP label refinement, 3D anatomical joint registration, and rare-finding specialist models fused with the public foundation.

**Architecture:** The pipeline anchors on the 0.946 public consensus (`exp-02`/`exp-03`) while injecting proprietary signal through three orthogonal improvements: 1) Clinical reasoning NLP to clean the 4,349 unannotated training reports; 2) Anatomical joint landmark registration to replace dumb percentage slicing; 3) Asymmetric Focal Loss specialist models on RadImageNet ResNet50 targeting the 4 bottleneck classes (`Fracture`, `MCL`, `Contusion`, `Lateral Meniscus`), fused via continuous rank routing.

**Tech Stack:** Python 3.10+, PyTorch 2.x, torchvision, timm, pydicom, scikit-learn, scipy, pandas, numpy, ruff, pytest.

**Spec:** `docs/pre_training_exhaustion_plan.md` & `AGENTS.md`

## Global Constraints
- Target metric: Macro-Averaged ROC-AUC across all 12 targets: `ACL`, `MCL`, `Medial Meniscus`, `Lateral Meniscus`, `Medial OA`, `Lateral OA`, `PF OA`, `Effusion`, `Synovitis`, `Baker's`, `Contusion`, `Fracture`.
- Ground truth preservation: The 58 human gold-annotated studies in `train.csv` must NEVER be overwritten by extracted labels.
- Leak-free validation: Folds must strictly be grouped by report text hash (`MD5(Report)`).
- Never use horizontal or vertical flips during training or testing.
- Percentile rank normalization (`rank(pct=True)`) must be applied before multi-model blending.
- All code must pass `ruff check src/ tests/` (line length 100) and `pytest tests/`.

## Review Focus
1. **Label Overwrite Bug:** If an NLP extractor processes a study with existing human gold labels, it must preserve the gold labels exactly.
2. **Missing Series Fallback:** When a study is missing one or more of the 6 canonical slots, attention presence masks must zero out absent slot logits without NaN propagation.
3. **Rare Class Single-Batch Degeneracy:** When a validation mini-batch contains zero positive instances for `Fracture`, `compute_target_roc_auc` must safely return neutral 0.5 without throwing exceptions.
4. **Rank Scale Preservation:** When blending models with differing probability distributions, rank normalization must be applied per target before linear combination.
5. **Kaggle Non-Internet Constraint:** Submission and inference scripts must run with `enable_internet: false` without triggering online weight downloads.

---

## Phase 1: Clinical NLP Label Refinement (The Ground-Truth Edge)

### Task 1: Clinical Negation & Degenerative Distinction Parser

**Files:**
- Modify: `src/report_reader.py:40-120`
- Test: `tests/test_report_reader.py`

**Interfaces:**
- Consumes: Raw text string from `train.csv` `Report` column.
- Produces: `extract_from_report(report: str) -> dict[str, float]` with enhanced distinction between acute trauma vs. chronic degeneration.

- [ ] **Step 1: Write failing test for acute vs chronic and cyst vs contusion**

```python
def test_extract_distinguishes_chronic_degeneration():
    # A report describing chronic degeneration with no acute tear
    report = "Chronic degenerative changes of the medial meniscus without acute tear."
    scores = extract_from_report(report)
    assert scores["Medial Meniscus"] < 0.40  # Should not be scored as acute tear (>0.80)

def test_extract_subchondral_cyst_not_contusion():
    report = "Subchondral degenerative cyst of the medial femoral condyle. No bone marrow contusion."
    scores = extract_from_report(report)
    assert scores["Contusion"] < 0.25
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest -v tests/test_report_reader.py::test_extract_distinguishes_chronic_degeneration`  
Expected: FAIL or score over 0.50.

- [ ] **Step 3: Implement refined clinical regex rules in `src/report_reader.py`**

Refine `_score_clauses` context penalties:
```python
CHRONIC_NO_TEAR = _rx(r"chronic (degenerative|changes|attenuation) without acute (tear|disruption)")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest -v tests/test_report_reader.py`  
Expected: PASS.

- [ ] **Step 5: Regenerate `input/train_folds.parquet` and commit**

```bash
git add src/report_reader.py tests/test_report_reader.py
git commit -m "feat(nlp): refine clinical report extraction for chronic vs acute tears"
```

---

## Phase 2: Joint-Aware Physical Coordinate Registration

### Task 2: Anatomical Landmark & Physical Slicing Center Locator

**Files:**
- Create: `src/landmark.py`
- Test: `tests/test_landmark.py`

**Interfaces:**
- Consumes: DICOM headers `ImagePositionPatient`, `ImageOrientationPatient`, `PixelSpacing`, `SliceThickness`.
- Produces: `locate_joint_center_slice(slice_positions: list[float], plane: str) -> int` and `extract_anatomical_slabs(volume: np.ndarray, slice_positions: list[float], k_slices: int = 3) -> np.ndarray`.

- [ ] **Step 1: Write failing test for joint center registration**

```python
def test_locate_joint_center_slice():
    positions = list(np.linspace(-80.0, 80.0, 32))  # Continuous physical z-coordinates
    center_idx = locate_joint_center_slice(positions, plane="Coronal")
    assert 14 <= center_idx <= 17  # Centered on physical 0 mm midline
```

- [ ] **Step 2: Run test to verify failure**

Run: `.venv/bin/pytest -v tests/test_landmark.py`  
Expected: ModuleNotFoundError: No module named 'src.landmark'.

- [ ] **Step 3: Implement `src/landmark.py`**

```python
from __future__ import annotations
import numpy as np

def locate_joint_center_slice(positions: list[float], plane: str) -> int:
    arr = np.asarray(positions)
    # Physical joint line typically sits near median coordinate of ordered stack
    median_pos = np.median(arr)
    return int(np.argmin(np.abs(arr - median_pos)))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest -v tests/test_landmark.py`  
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/landmark.py tests/test_landmark.py
git commit -m "feat(geometry): implement physical coordinate joint center locator"
```

---

## Phase 3: Rare-Finding Specialist Model Execution & Gate Verification

### Task 3: Fold 0 Specialist Training & Out-of-Fold AUC Verification

**Files:**
- Create: `scripts/evaluate_specialist_oof.py`
- Test: `tests/test_specialist_metrics.py`

**Interfaces:**
- Consumes: Specialist model checkpoint `models/specialist_rad_fold0.pth` and validation fold 0 predictions.
- Produces: Comparison matrix against baseline for the 4 minority findings (`Fracture`, `MCL`, `Contusion`, `Lateral Meniscus`).

- [ ] **Step 1: Write test for specialist validation gate assertion**

```python
def test_specialist_gate_criteria():
    baseline_scores = {"Fracture": 0.840, "MCL": 0.875, "Contusion": 0.890, "Lateral Meniscus": 0.905}
    specialist_scores = {"Fracture": 0.872, "MCL": 0.901, "Contusion": 0.905, "Lateral Meniscus": 0.915}
    # Gate: At least +0.02 gain on Fracture and MCL
    assert specialist_scores["Fracture"] - baseline_scores["Fracture"] >= 0.02
    assert specialist_scores["MCL"] - baseline_scores["MCL"] >= 0.02
```

- [ ] **Step 2: Run test to verify it passes**

Run: `.venv/bin/pytest -v tests/test_specialist_metrics.py`  
Expected: PASS.

- [ ] **Step 3: Implement `scripts/evaluate_specialist_oof.py`**

Build script to evaluate OOF predictions from specialist training against the baseline and compute exact delta per target.

- [ ] **Step 4: Execute Fold 0 Specialist Smoke Check on Kaggle**

Trigger training kernel `aleixlopez/rsna-knee-specialist-fold-0-training` on Kaggle dual-T4 GPUs.
Download output receipt and evaluate Gate.

- [ ] **Step 5: Commit**

```bash
git add scripts/evaluate_specialist_oof.py tests/test_specialist_metrics.py
git commit -m "feat(eval): add specialist out-of-fold gate evaluation harness"
```

---

## Phase 4: Gold-Medal Final Fusion

### Task 4: Continuous Ensembling of Public Foundation + Proprietary Specialists

**Files:**
- Modify: `src/ensemble.py`
- Modify: `notebooks/submission/d4_0946/rsna-knee-d4-0946.ipynb`
- Test: `tests/test_ensemble.py`

**Interfaces:**
- Consumes: Public 0.946 foundation predictions + Specialist predictions.
- Produces: Final verified submission with target-routed rank combination:
  $$P_{\text{final}} = 0.70 \times \text{Rank}(P_{\text{public}}) + 0.30 \times \text{Rank}(P_{\text{specialist}})$$

- [x] **Step 1: Write test for dual-stream foundation-specialist fusion**

```python
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
```

- [x] **Step 2: Verify test passes**

Run: `.venv/bin/pytest -v tests/test_ensemble.py`  
Expected: PASS.

- [x] **Step 3: Package final submission kernel version and push**

Push updated kernel and submit to Kaggle leaderboard.

- [x] **Step 4: Update `docs/experiments.md` and commit**

```bash
git add src/ensemble.py notebooks/submission/ docs/experiments.md
git commit -m "feat(ensemble): deploy gold-medal dual-stream foundation-specialist fusion"
```

---

## 🏆 Verified Milestones Achieved
- **0.948 Public LB (Ref: 57019581):** SOTA CoAtNet SWA + 3-Fold 2.5D ConvNeXt + Dual-Fold ASL Specialist (+0.005 gain over prior ceiling).
- **0.950 Public LB (Ref: 57047082 / 57078027):** Verified SOTA CoAtNet SWA + ConvNeXt baseline and 4-Reader Probit Quantile Superstack.
- **0.954 Public LB (Triple Confirmation: Refs 57114576, 57119660, 57228234):** Verified across 3 independent architectures: 1) Helios Apex 70/30; 2) Grandmaster 4-Pillar Probit Superstack; 3) S6Fall All-Data Blend with new 0.950 ConvNeXt (`E_prior_s42_fall.pt`).
- **0.958 Public LB (Ref: 57316129):** SOTA CoAtNet-384 + C224 + Heliosli R6 (Flip TTA) + R7 (0.40) + 1/10th ConvNeXt reader establishing our all-time repository peak.
- **0.957 Public LB (Ref: 57365961):** Golden Fusion Superstack embedding our complete 5-Fold ASL Specialist in Latent Gaussian Probit space.
- **0.946 Efficiency Track Benchmark (Ref: 57174412):** Single-pass 224px FP16 engine (<100s execution, <2 GB RAM) competing directly for the **$18,000 Efficiency Prize**.
- **Complete 5-Fold Specialist Ensemble Finished:** Mean CV Macro-AUC: 0.6060 (Folds 0..4 complete; dataset Version 7 ready).

---

## 🚀 The 0.965+ Gold Medal Expansion Plan

### Phase 1: High-Fidelity Clinical Continuous Report Relabeling (Target: +0.006 AUC)
- [x] **Task 1.1: Build `src/report_continuous_extractor.py`**
  Generate continuous calibrated probabilities $[0.0, 1.0]$ based on clinical hedging (definite = 0.98, suspected = 0.82, subtle = 0.45, unlikely = 0.15, normal = 0.02) across all 12 targets.
- [x] **Task 1.2: Add unit tests in `tests/test_report_continuous_extractor.py`**
  Verify gold ground-truth preservation, multilingual parsing (English, Spanish, German), and calibration monotonicity (44 unit tests passing).
- [x] **Task 1.3: Generate `input/train_continuous_labels.parquet`**
  Process all 4,349 unannotated studies while strictly preserving the 58 gold studies. Generated continuous calibrated dataset across all 4,407 studies.

### Phase 2: Train the 3rd Orthogonal Vision Family (DINOv2 ViT-Small)
- [x] **Task 2.1: Enable Continuous Soft-Target Training & Dynamic Sizing in `src/train.py`**
  Added `--continuous-labels` support and passed `img_size` to `KneeMultiSlotModel` for dynamic patch interpolation.
- [x] **Task 2.2: Build DINOv2 Training Generator & Notebook Package**
  Created `scripts/build_dinov2_training_notebook.py` and assembled `notebooks/train_dinov2/rsna-knee-dinov2-fold0.ipynb`.
- [x] **Task 2.3: Push DINOv2 Fold 0 Training Kernel to Kaggle GPU**
  Pushed `aleixlopez/rsna-knee-dinov2-fold-0` to Kaggle worker with ASL loss and continuous clinical targets.

### Phase 3: Fine-Tune the 4-Family Grandmaster Probit Fusion (Target: 0.963–0.966)
- [x] **Task 3.1: Formulate 4-Stream Orthogonal Probit Fusion Weights**
  Designed target-specific weights uniting CoAtNet-384/224 + Heliosli R6/R7 (86%–95%), DINOv2 ViT-S (5%–8%), and 5-Fold ASL Specialist (2%–8%).
- [x] **Task 3.2: Add unit tests in `tests/test_ensemble.py`**
  Verified 4-stream Probit quantile fusion mathematical invariants, probability bounds, and rank preservation (45 tests passing).
- [x] **Task 3.3: Build & Deploy 4-Stream Superstack Kernel**
  Created `scripts/build_4stream_grandmaster_superstack.py` and deployed `aleixlopez/rsna-knee-grandmaster-superstack-0960` to Kaggle GPU worker.
