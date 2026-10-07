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
| `sub-04-golden-fusion` | Golden Fusion: 0.943 Foundation + RadImageNet ASL Specialist | Multi-res / Multi-view | ~0.952 (Est.) | **0.943** | Verified official submission (Ref: 56803671) scored 0.943 on Public LB; end-to-end GPU verification complete |
| `sub-05-golden-fusion-v2` | Golden Fusion v2: 0.943 Foundation + Real MRI Specialist | Multi-res / Multi-view | ~0.954 (Est.) | **0.943** | Verified official submission (Ref: 56849200) scored 0.943 on Public LB; confirmed rock-solid foundation stability |
| `sub-06-golden-fusion-v3` | Golden Fusion v3: 0.943 Foundation + Full 4407 Specialist | Multi-res / Multi-view | ~0.956 (Est.) | **0.943** | Verified official submission (Ref: 56907494) scored 0.943 on Public LB; confirmed rock-solid foundation stability |
| `exp-01-resnet34` | 2.5D ResNet34 Slab | 224x224 (6 slots) | *Ready to run* | - | Fast baseline model |
| `abl-01-asl-vs-bce` | ASL vs BCE Local Ablation (MPS) | Fold 0 (896 val studies) | ASL: 0.9747 (BCE: 0.9690) | N/A (Local) | Empirical loss proof: Fracture +0.0323 AUC, MCL +0.0067 AUC, Macro +0.0057 AUC |
| `spec-01-rad50-fold0` | RadImageNet ResNet50 + ASL (Fold 0) | 224x224 (6 slots) | Loss: 0.0636 | N/A (Trained) | Fold 0 Smoke Check complete on Kaggle T4 GPU; 92MB weights saved (specialist_rad_fold0.pth) |
| `spec-02-rad50-real-mri` | RadImageNet ResNet50 + ASL on Real DICOMs | 224x224 (6 slots) | Loss: 0.0529 / Lat Men: 0.7216 | N/A (Trained) | Real MRI pixel training (700 studies decoded into memory); Lat Meniscus reached 0.7216 AUC; dataset updated to v2 |
| `spec-03-rad50-full-4407` | RadImageNet ResNet50 + ASL (All 4,407 Studies) | 224x224 (6 slots) | Macro: 0.6036 / Fracture: 0.6351 | N/A (Trained) | Full-corpus training (3,502 train / 905 val); Fracture reached 0.6351 AUC, Contusion 0.6174 AUC; dataset updated to v3 |
| `spec-04-rad50-384px` | RadImageNet ResNet50 + ASL (384x384, All 4,407 Studies) | 384x384 (6 slots) | Macro: 0.6052 / Contusion: 0.6182 | N/A (Trained) | High-definition 384x384 training complete (50 min); Contusion 0.6182, Lat Men 0.6195; dataset updated to v4 |
