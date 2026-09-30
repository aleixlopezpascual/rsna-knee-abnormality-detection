# Agent Instructions: RSNA Knee Abnormalities Detection

This workspace is set up for the **RSNA Knee Abnormalities Detection** competition. Adhere to these instructions for all development, experiments, and submissions.

---

## 🎯 1. Competition Anchor & Metric

- **Objective:** Predict confidence scores in $[0, 1]$ for 12 binary knee abnormality targets.
- **Evaluation Metric:** Macro-Averaged ROC-AUC across all 12 targets:
  `ACL`, `MCL`, `Medial Meniscus`, `Lateral Meniscus`, `Medial OA`, `Lateral OA`, `PF OA`, `Effusion`, `Synovitis`, `Baker's`, `Contusion`, `Fracture`.
- **Authoritative Benchmark:** Always evaluate models using `src.metrics.compute_competition_metric(y_true, y_pred)`.
- **Validation Discipline:** A submission is never made to Kaggle without logging the local 5-fold CV score in `docs/experiments.md`.

---

## 📂 2. File & Directory Layout

- **`src/`**: Core reusable library code:
  - `src/config.py`: Target list, study ID column, random seeds, paths.
  - `src/metrics.py`: Official competition metric computation.
  - `src/dataset.py`: MRI slice loaders, DICOM parsers, and fold generators.
  - `src/models/`: Backbone neural architectures (2.5D CNNs, ViTs, Multi-view aggregators).
- **`input/`**: Raw competition files and DICOM series (`test_series/`, `train.csv`, `test.csv`).
- **`notebooks/`**: Exploratory analyses and reference competitor scripts (`notebooks/public_baselines/`).
- **`oof/`**: Out-of-fold prediction matrices for ensembling and error analysis.
- **`submissions/`**: Output submission files (`submission_<experiment_id>_<cv_score>.csv`).
- **`docs/`**: `competition-brief.md`, `experiments.md`, and harvested `intel/`.

---

## 🧪 3. Code Standards & Testing

- Python 3.10+ with type hints.
- Run `pytest` on all changes to ensure metric calculation and data transforms remain correct.
- Format and lint with `ruff check .`.
