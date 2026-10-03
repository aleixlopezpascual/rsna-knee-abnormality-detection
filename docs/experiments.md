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
| `sub-02-d4-0946` | 4-Pillar SOTA: DINOv2 + RadImageNet + Raptor + CoAtNet | Multi-res / Multi-view | ~0.944 | **0.943** | Verified official submission (Ref: 56749768) scored 0.943 on Public LB |
| `sub-03-d4-opt-k112` | 4-Pillar SOTA + Densified k_eval=112 + Refined Routing | Multi-res / Multi-view | ~0.947 (Est.) | **0.943** | Verified official submission (Ref: 56750202) scored 0.943 on Public LB |
| `exp-01-resnet34` | 2.5D ResNet34 Slab | 224x224 (6 slots) | *Ready to run* | - | Fast baseline model |
| `abl-01-asl-vs-bce` | ASL vs BCE Local Ablation (MPS) | Fold 0 (896 val studies) | ASL: 0.9747 (BCE: 0.9690) | N/A (Local) | Empirical loss proof: Fracture +0.0323 AUC, MCL +0.0067 AUC, Macro +0.0057 AUC |
| `spec-01-rad50-fold0` | RadImageNet ResNet50 + ASL (Fold 0) | 224x224 (6 slots) | Loss: 0.0636 | N/A (Trained) | Fold 0 Smoke Check complete on Kaggle T4 GPU; 92MB weights saved (specialist_rad_fold0.pth) |
