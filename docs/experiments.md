# Experiment Ledger: RSNA Knee Abnormalities Detection

All experiments must track local 5-fold CV before submitting to the Kaggle public leaderboard.

- **Primary Metric:** Macro-Averaged ROC-AUC across 12 targets (`src/metrics.py`)
- **Folds Protocol:** 5-Fold Multilabel Stratified Split on `StudyInstanceUID`

---

## Leaderboard & Experiment History

| Exp ID | Model Architecture | Resolution / Views | Local CV (Macro AUC) | Public LB | Notes / Insights |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `baseline-00` | *Metric Verification* | N/A | 1.000 (synthetic) | - | Metric unit test passed (5/5) |
| `exp-01-resnet34` | 2.5D ResNet34 Slab | 224x224 (3 planes) | *Pending* | - | Fast baseline model |
