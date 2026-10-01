# Experiment Ledger: RSNA Knee Abnormalities Detection

All experiments must track local 5-fold CV before submitting to the Kaggle public leaderboard.

- **Primary Metric:** Macro-Averaged ROC-AUC across 12 targets (`src/metrics.py`)
- **Folds Protocol:** 5-Fold Multilabel Stratified Split on `StudyInstanceUID`

---

## Leaderboard & Experiment History

| Exp ID | Model Architecture | Resolution / Views | Local CV (Macro AUC) | Public LB | Notes / Insights |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `baseline-00` | *Metric Verification* | N/A | 1.000 (synthetic) | - | Metric unit test passed (5/5) |
| `exp-01-pipeline` | 2.5D Multi-Slot Bag + SlotHead | 224x224 (6 slots) | Verified | - | Pipeline complete: Report NLP extractor, 5-fold grouped split, SlotHead attention, train/infer verified (18/18 tests passed) |
| `sub-01-baseline` | 2.5D Multi-Slot Bag Ensemble (DINOv2) | 336x336 (6 slots) | 0.8870 | **0.891** | Verified official submission (Ref: 56748754) scored 0.891 on Public LB |
| `sub-02-d4-0946` | 4-Pillar SOTA: DINOv2 + RadImageNet + Raptor + CoAtNet | Multi-res / Multi-view | ~0.944 | *Scoring (Ref: 56749768)* | High-water mark: 4-way CoAtNet consensus, RadImageNet ResNet50, Raptor k_eval=94 |
| `sub-03-d4-opt-k112` | 4-Pillar SOTA + Densified k_eval=112 + Refined Routing | Multi-res / Multi-view | ~0.947 (Est.) | *Running v2* | Zero-training gain: k_eval densified to 112, MCL=0.75, Medial OA=0.72, Fracture=0.78, Contusion=0.65 |
| `exp-01-resnet34` | 2.5D ResNet34 Slab | 224x224 (6 slots) | *Ready to run* | - | Fast baseline model |
