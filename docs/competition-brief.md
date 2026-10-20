# RSNA Knee Abnormalities Detection: Competition Brief

- **Competition:** RSNA Knee Abnormalities Detection (`rsna-knee-abnormality-detection`)
- **Host:** Radiological Society of North America (RSNA)
- **Prize Pool:** $77,000 USD (Main Leaderboard + Efficiency Track)
- **Deadline:** October 22, 2026

---

## 1. Problem Statement & Modality

The challenge is to develop machine learning models that detect **12 clinically important knee abnormalities** from multi-sequence knee MRI studies.

- **Modality:** Multimodal medical imaging.
  - Visual: Multi-sequence 3D/volumetric knee MRI DICOM series (`test_series/`, `train_series/` containing Sagittal, Coronal, and Axial planes).
  - Text: Training data includes paired radiology diagnostic reports.
- **Input Key:** `StudyInstanceUID`
- **Target Attributes (12 Abnormalities):**
  1. `ACL` (Anterior Cruciate Ligament tear)
  2. `MCL` (Medial Collateral Ligament tear)
  3. `Medial Meniscus` (Tear/degeneration)
  4. `Lateral Meniscus` (Tear/degeneration)
  5. `Medial OA` (Osteoarthritis)
  6. `Lateral OA` (Osteoarthritis)
  7. `PF OA` (Patellofemoral Osteoarthritis)
  8. `Effusion` (Joint fluid accumulation)
  9. `Synovitis` (Inflammation of synovial membrane)
  10. `Baker's` (Baker's cyst)
  11. `Contusion` (Bone bruise)
  12. `Fracture` (Bone fracture)

---

## 2. Official Evaluation Metric

Submissions are scored on **Macro-Averaged ROC-AUC** across the 12 binary targets:
$$
\text{Final Score} = \frac{1}{12} \sum_{i=1}^{12} \text{ROC-AUC}_i
$$

### Metric Nuances & Stability:
- If a fold or batch has only negative samples for an extremely rare class (e.g. `Fracture`), standard `roc_auc_score` raises a ValueError. The local metric implementation must handle single-class targets gracefully or guarantee stratified folds with representation across all 12 classes.
- Continuous confidence scores in $[0, 1]$ are required (not thresholded binary $0/1$ predictions).

---

## 3. Cross-Validation Scheme

- **Split Protocol:** **5-Fold Multilabel Stratified Split** grouped by `StudyInstanceUID`.
- **Rationale:** Because each study has 12 simultaneous binary labels with varying frequencies (e.g., `Effusion` is very common, whereas `Fracture` and `MCL` are relatively sparse), simple random split risks fold imbalance. Using iterative multilabel stratification (`iterative-stratification` / `MultilabelStratifiedKFold`) ensures balanced ROC-AUC surfaces across all folds.
- **Golden Rule:** Never trust public LB jumps without CV correlation. Local CV is our authoritative anchor.

---

## 4. Key Community Traps & Architectural Insights

1. **The Train vs. Test Modality Trap:**
   - Training studies include rich text radiology reports, but test inference will only have raw DICOM imaging series!
   - *Strategy:* Use text reports during training via cross-modal contrastive learning or distillation (e.g., train a text-teacher LLM/DeBERTa to produce soft labels or guide visual embeddings), but ensure the test inference pipeline depends strictly on the MRI images.
2. **247 GB Dataset Sizing & Image Preprocessing:**
   - The raw DICOM files total 247 GB. Storing and reading raw DICOMs during epoch loops causes severe I/O bottlenecks.
   - *Best Practice:* Preprocess and cache 2.5D multi-plane slab arrays (Sagittal PD/T2, Coronal, Axial) at $224 \times 224$ or $256 \times 256$ in lightweight `.npy` or `.parquet` format.
3. **Series View Routing:**
   - Knee MRI abnormalities exhibit specific canonical orientations:
     - `ACL`, `PCL`: Best visualized on Sagittal views.
     - `MCL`, `LCL`, Meniscal bodies: Best visualized on Coronal views.
     - `Patellofemoral OA`, `Synovitis`: Best visualized on Axial views.
   - Routing distinct series views to dedicated visual backbone branches or view-attention pooling significantly outperforms treating all slices uniformly.

---

## 5. Phase 1 Execution Plan

1. **Metric & Schema Verification:** Implement and unit test the 12-target macro ROC-AUC in `src/metrics.py`.
2. **Dataset Preprocessing Pipeline:** Build a generator in `src/dataset.py` that parses DICOM metadata, selects key slices/slabs per view (Sagittal, Coronal, Axial), and resizes them.
3. **Fast Baseline Model:** Train a 2.5D ResNet34 / ConvNeXt backbone with multi-label binary cross-entropy loss, producing fold 0 OOF predictions and a verified submission.

---

## 6. SOTA Benchmarks & Official Submissions

- **Current Public Leaderboard Peak (2026-10-20 snapshot):** **0.958 Macro ROC-AUC** (`exp-18`, Ref: `57316129`).
- **Current final-selection plan:** Slot 1 is `exp-18` (0.958); Slot 2 is one of `exp-19` / `exp-20` (both 0.957) as a main-track hedge. The `exp-15` efficiency submission (0.946) is not the default second selection because its public Efficiency Leaderboard rank was 174/6,159 and only three Efficiency Prizes are awarded. See [`docs/final-submission-selection.md`](final-submission-selection.md) for the dated snapshot, rationale, and approval status.
- **5-Fold Specialist Ensemble:** 100% complete across all 4,407 studies (Folds 0..4, Mean CV: **0.6060**; dataset Version 7).
