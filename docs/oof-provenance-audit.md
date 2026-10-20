# Kaggle OOF Provenance Audit

**As of:** 2026-10-21 00:04 CEST
**Scope:** Trace the Kaggle datasets attached to the current top submissions and locate reusable, provenance-backed out-of-fold (OOF) predictions.
**Status:** Static inspection/download of public metadata and small prediction artifacts only. No downloaded notebook was executed; no submission was made. The inspected Kaggle files were kept in temporary scratch, not copied into this repository.

## Executive finding

Public OOF artifacts do exist, but **none found so far is an OOF cache for the exact 0.958 Exp-18 anchor or the anchor-plus-ASL-specialist stack**. The best relevant discoveries are:

1. A 4,407-study DINOv2-Small prediction cache in [`pilkwang/rsna-knee-weights`](https://www.kaggle.com/datasets/pilkwang/rsna-knee-weights). Its manifest lists 20 DINOv2-Small members across five folds and four seeds. Its OOF-shaped array covers all local training IDs and marks exactly the 58 official gold studies, but the NPZ has no per-row fold field or separate OOF-generation receipt. Treat it as a **candidate OOF artifact pending provenance verification**, not as verified evidence for the current anchor.
2. A separately published, explicitly cross-fitted, report-hash-grouped 4,407-row table from [Prvsiyan's V52 notebook](https://www.kaggle.com/code/prvsiyan/rsna-knee-read-the-report-then-the-knee), published as [`prvsiyan/rsna-knee-v52-oof-receipt-20260810`](https://www.kaggle.com/datasets/prvsiyan/rsna-knee-v52-oof-receipt-20260810). It is a different model, not a component of our current stack.
3. A 5-fold EfficientNet-B3 OOF prediction artifact for only the 58 gold-labeled studies from [`nekkon/rsna-knee-image-oof-gold`](https://www.kaggle.com/datasets/nekkon/rsna-knee-image-oof-gold). It is also a different model and covers only the gold subset.

These artifacts may support separate research or comparisons on eligible studies. They do **not** by themselves enable a CV-gated blend evaluation of Exp-18 versus our ASL specialist: we still need fold-safe predictions for the exact anchor streams and specialist, with provenance that proves each prediction is from a model that did not train on that study.

## 1. Current best submission lineage

The current main-score anchor is **Exp-18, submission `57316129`, public AUC 0.958**. Its local Kaggle metadata is [`notebooks/submission/apex_0957_sota/kernel-metadata.json`](../notebooks/submission/apex_0957_sota/kernel-metadata.json). It attaches these five public datasets:

| Kaggle dataset | Files/source characteristics found | OOF status for this audit |
|---|---|---|
| [`nartaa/rsna-knee-publication-swa-weights-20261007`](https://www.kaggle.com/datasets/nartaa/rsna-knee-publication-swa-weights-20261007) | Two CoAtNet SWA checkpoints plus a checkpoint manifest and README. The dataset description says it distributes learned checkpoints and aggregate metadata, **not** subject-level predictions or cached volumes. | No OOF prediction table found. |
| [`heliosli/rsna-orthofoundation-fold0-epoch7-20261004`](https://www.kaggle.com/datasets/heliosli/rsna-orthofoundation-fold0-epoch7-20261004) | Large code/weight package (837 listed files), including `anatomy.pt` and extensive training/runtime code. The complete file listing was paged and filtered for OOF/prediction/validation-style filenames; no prediction cache was found. | No OOF table found in the dataset files. |
| [`rostigor/rsna-knee-r6-fe055406`](https://www.kaggle.com/datasets/rostigor/rsna-knee-r6-fe055406) | Four R6 fold checkpoints plus manifests/configuration JSON. | No OOF prediction file in the listed contents. |
| [`heliosli/rsna-knee-compact-r7cnxfsss`](https://www.kaggle.com/datasets/heliosli/rsna-knee-compact-r7cnxfsss) | Four R7 fold checkpoints plus manifests/configuration JSON. | No OOF prediction file in the listed contents. |
| [`goodpjw2008/rsna-knee-2-5d-convnext-reader`](https://www.kaggle.com/datasets/goodpjw2008/rsna-knee-2-5d-convnext-reader) | Three ConvNeXt fold checkpoints and source files, including `oof.py`, `train.py`, and `infer.py`. | Contains OOF-related **code**, but no saved OOF prediction table. |

The higher-scoring Grandmaster notebook (`aleixlopez/rsna-knee-grandmaster-superstack-0960`, used for the 0.957 submission `57410300`) attaches the Exp-18 sources plus the five-fold specialist weights and Pilkwang’s DINOv2 dataset. Its latest pulled source constructs a `vit_base_patch14_dinov2` model, searches that dataset for the first `.pt` file, and loads it; it does **not** read `oof.npz`. The dataset manifest describes its 20 checkpoints as `facebook/dinov2-small`, 336 px. Therefore the public DINOv2-Small OOF cache is not automatically an OOF validation artifact for the Grandmaster notebook’s DINO branch either. The notebook's “loaded” log line is not a checkpoint-compatibility receipt.

The ASL specialist dataset [`aleixlopez/rsna-knee-specialist-weights`](https://www.kaggle.com/datasets/aleixlopez/rsna-knee-specialist-weights) contains five fold checkpoints and training logs, but no saved per-study OOF prediction file. The two local files under `oof/` are also unusable: each has four rows, and none of the IDs match `input/train_folds.parquet`.

## 2. Public OOF artifacts located

### A. Pilkwang DINOv2-Small cache — candidate, not yet provenance-approved

- **Source notebook:** [Pilkwang — RSNA Knee baseline v1](https://www.kaggle.com/code/pilkwang/rsna-knee-baseline-v1), whose public page reports a 0.891 public score.
- **Dataset:** [`pilkwang/rsna-knee-weights`](https://www.kaggle.com/datasets/pilkwang/rsna-knee-weights), CC0-1.0 according to its Kaggle page.
- **Dataset evidence:** `manifest.json` lists 20 members. The member configurations identify DINOv2-Small (`facebook/dinov2-small`), with fold IDs 0–4 repeated across four source runs/seeds.
- **`oof.npz` contents verified:** keys `ids` (4,407), `pred` (4,407 × 12), `y_derived` (4,407 × 12), `gold_mask` (4,407), and `targets` (12). IDs are unique and their set exactly matches `input/train.csv` / `input/train_folds.parquet`. `gold_mask` selects exactly the 58 studies with official labels.
- **Important limitations:** the NPZ has no per-row fold column; the dataset card has no OOF-generation description; and `y_derived` does not match the labels in this repository's `train_folds.parquet` (it is a different weak-label source). The manifest's fold-member list is useful evidence, but it does not by itself prove which held-out checkpoint produced every row. Verify this before using the artifact as CV evidence.
- **Stack compatibility:** this is a DINOv2-Small stream, not the CoAtNet/Helios/R6/R7/ConvNeXt Exp-18 anchor. The Grandmaster notebook attaches the dataset but does not consume the OOF file; its code attempts inference from a single `.pt` checkpoint instead of using the complete OOF/member ensemble.
- **Temporary-file SHA-256:** `oof.npz` = `62d47ba4c0c8347b5b24e7fd2aa517aae0fd6d4656fd5d829fd3a50f0159909c`; `merge_gain.npz` = `d8af61726d9ec07abba19cabfa9c548de05bdb11206c9442a3d72ede737eb004`. `merge_gain.npz` has 4,407 aligned rows and arrays `ours`, `imported`, `y`, `gold_mask`, but its stream semantics/provenance were not established; do not treat it as anchor OOF.

### B. Prvsiyan V52 — explicit full-cohort OOF for a different model

- **Source kernel/version:** [`prvsiyan/rsna-knee-read-the-report-then-the-knee`, V52](https://www.kaggle.com/code/prvsiyan/rsna-knee-read-the-report-then-the-knee).
- **Dataset:** [`prvsiyan/rsna-knee-v52-oof-receipt-20260810`](https://www.kaggle.com/datasets/prvsiyan/rsna-knee-v52-oof-receipt-20260810), CC BY-NC-SA 4.0 according to the Kaggle download response.
- **Provenance:** `PROVENANCE.md` identifies the source notebook version and terminal output `rsna_rad_v52/v52_oof.csv`, calls the table cross-fitted and report-hash-grouped, and records SHA-256 `43bc2cf07ce924e2d02e3240c31638130c3275882caaed77e4368d39dbf465ff`.
- **Local checks:** 4,407 rows, 4,407 unique IDs, exact match to the competition training IDs, no duplicate IDs, 12 prediction columns, fold counts `{0: 882, 1: 882, 2: 881, 3: 881, 4: 881}`, and an `is_gold` set exactly matching the 58 official gold studies. Predictions are finite and in `[0, 1]`.
- **Limitations:** it has no ground-truth target columns and is not part of our current stack. Join official labels where available; any reuse must respect its license and source-model terms.

### C. NeKKon gold-only OOF — separate image model, 58 rows

- **Dataset:** [`nekkon/rsna-knee-image-oof-gold`](https://www.kaggle.com/datasets/nekkon/rsna-knee-image-oof-gold), CC0-1.0 according to the Kaggle download response.
- Its README describes `pred_b3_gold_5folds.npy` as five-fold EfficientNet-B3 multi-slot MIL predictions for the 58 annotated studies, which were not used in training. The model was trained using text-derived weak labels from 929 of the 4,407 studies due to disk limits.
- `gold_order.csv` was checked and its 58 IDs match the official labeled IDs in `input/train.csv` in order. The prediction `.npy` itself was not downloaded or independently checked in this audit.
- **Limitations:** gold-subset-only evidence for a different model; not full-cohort OOF for the current anchor or ASL specialist.

## 3. What this does—and does not—unlock

We have found public OOF data for other model lineages, not the exact OOF stream needed to compare the current `0.958` anchor against the ASL specialist. In particular, a public OOF cache cannot validate a different model merely because both use DINOv2 or because the cache covers the same study IDs.

For an anchor-versus-specialist CV test, we still need:

1. **Anchor predictions:** one fold-safe prediction per study from each exact anchor stream, or a documented independent model that did not train on the corresponding study. Verify model/checkpoint provenance and preprocessing.
2. **Specialist predictions:** one held-out prediction per study from the appropriate specialist fold checkpoint. The five checkpoints exist, but their per-study OOF output is not saved.
3. **Auditable alignment:** predictions keyed by `StudyInstanceUID`, complete coverage, duplicates/missing-ID checks, fold provenance, and labels from the official training table (only the 58 studies have human ground truth; the remaining rows have report-derived/weak targets).
4. **Honest weight selection:** choose blend weights on development folds and confirm on separate folds; do not optimize and report performance on the same OOF rows.

Until the exact anchor OOF is recovered or generated from fold-safe checkpoints, the public OOF files above are references or potential alternative streams—not evidence that a new blend improves Exp-18. No blend was evaluated or promoted in this audit.

## Source links

- [Exp-18 Apex kernel](https://www.kaggle.com/code/aleixlopez/rsna-knee-apex-0957-sota)
- [Grandmaster Superstack kernel](https://www.kaggle.com/code/aleixlopez/rsna-knee-grandmaster-superstack-0960)
- [Pilkwang baseline notebook](https://www.kaggle.com/code/pilkwang/rsna-knee-baseline-v1)
- [Pilkwang weights / OOF dataset](https://www.kaggle.com/datasets/pilkwang/rsna-knee-weights)
- [Prvsiyan V52 OOF dataset](https://www.kaggle.com/datasets/prvsiyan/rsna-knee-v52-oof-receipt-20260810)
- [NeKKon gold OOF dataset](https://www.kaggle.com/datasets/nekkon/rsna-knee-image-oof-gold)
