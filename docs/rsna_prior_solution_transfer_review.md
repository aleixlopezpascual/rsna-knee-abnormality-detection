# Recent RSNA Winning Solutions: Transfer Audit & Experiment Backlog

**Research snapshot:** 2026-10-22 11:16 CEST

**Scope:** Source review of 15 top-place solution writeups across five RSNA Kaggle challenges (2025–2022), compared with the exact local Exp-18 notebook.

**Status:** Research and static inspection only. No downloaded notebook or model code was executed; no experiments, Kaggle kernels, submissions, or final-slot changes were made.

> **Bottom line:** Our strongest knee entry already has the common winning foundations—2.5D windows, sequence/target attention, physical-scale preprocessing, dense sampling, multi-resolution inference, anatomical mirror TTA, and target-specific rank blending. The strongest *unclosed* opportunity suggested by the writeups is better supervision and spatial conditioning: out-of-fold (OOF) detection of report-label errors, followed by a carefully isolated anatomy-guided crop/mask experiment. The exact current stack has no verified fold-excluded OOF cache, so these are hypotheses, not gains. Do not spend the remaining competition hours trying to train and submit one of these ideas: the public-LB-only Exp-22 already tied the anchor at displayed precision, and there is not time to validate a new pipeline credibly.

## 1. Decision context and review scope

The current live submission snapshot, queried with the Kaggle CLI at 2026-10-22 11:11 CEST, showed Exp-18 `57316129` **COMPLETE, 0.958 public**, Exp-22 micro-calibration `57478743` **COMPLETE, 0.958 public**, and Exp-19/Exp-20 `57365961`/`57410300` at **0.957**. Private scores were blank. Exp-22 was explicitly public-LB-only and is not CV evidence; its tie does not show a gain. Kaggle’s final-submission deadline is 2026-10-22 23:59 UTC, leaving under 15 hours at the research snapshot.[33] The full-data OOF audit says no exact fold-excluded predictions have been found for the Exp-18 stack or its anchor-plus-specialist blend (`docs/oof-provenance-audit.md`).

The selective review covered these 15 writeups: **2025 Intracranial Aneurysm** ranks 1–3; **2024 Lumbar Spine Degenerative Classification** ranks 1–6; **2023 Abdominal Trauma** ranks 1–2; **2023 Screening Mammography** ranks 1–3; and **2022 Cervical Spine Fracture** rank 1. This is a selected set of primary Kaggle writeups and linked author repositories—not an exhaustive review of every prize-winning team, notebook version, or discussion. Results below are author-reported unless specifically called an official placement; no source score is treated as directly comparable with knee macro ROC-AUC.

## 2. What the top writeups actually did

### 2025 — Intracranial Aneurysm Detection (multimodal CT/MR)

This is the newest completed RSNA Kaggle challenge in the review. Its inputs included CTA, MRA, and MRI; its metric was a weighted multilabel AUC that gave the overall-presence label substantially greater weight than each location label.[31]

**1st — “Location-Aware Aneurysm Detection via Vessel-ROI Masking” (tomoon33).** The author’s solution is a coarse-to-fine pipeline: three nnU-Net segmentation models first find a low-resolution vessel ROI, then refine vessel masks at higher resolution. A 3D classifier consumes the refined masks and uses location-masked feature pooling for 13 anatomical locations, global pooling for overall presence, and a location-aware Transformer across locations.[10][12][15]

The classifier also has an auxiliary sphere-localization task around annotated aneurysm coordinates. The author reports 5-fold multilabel stratification, exclusion of roughly 60 problematic series, four-fold inference averaging, and left/right flip TTA. The reported reduced-resolution ablation was 0.902 versus 0.916 at full resolution; removal of the auxiliary localization loss or segmentation model lowered the author’s ablation score. These were simplified, not re-tuned ablations, and they do not predict a knee gain.[10][15]

**2nd — “Fast 2D tri-axial ROI extraction + 3D Multi-Task Segmentation and Classification” (BraveCoWCoW).** Stage 1 uses 2D nnU-Net on nine slices (three per axis) to find a fast vascular ROI. Stage 2 resizes volumes to 224³ and trains 3D nnU-Net for vessel/aneurysm segmentation plus classification, with modality heads and rare-class oversampling. The team reports external TopCoW/TopBrain annotations, manual corrections, eight-view TTA with label swaps, and a two-fold ensemble on two T4s taking about nine hours. Their ablation table reports 0.844 public/0.813 private at baseline and 0.90035/0.86727 for the final ensemble; the prose gives private 0.86732, a small internal discrepancy. External labels and manual annotation are important dependencies, not ready-made knee inputs.[1]

**3rd — “3rd place solution” (BTYND).** This is a contrasting route: MIP-based sagittal/coronal views feed YOLO vessel-region detectors, whose boxes define fixed-size 3D ROIs. A 3D ResNet-18 applies class heads at every spatial feature-map location; top-N spatial scores are averaged into study predictions. The team reports 11 variants across crop sizes, resolutions, and strides, plus a separate 2.5D EfficientNet spacing regressor for missing DICOM tags. Its single-fold resolution comparison and final 0.86 public/0.84 private are author reports, not a fully specified CV ablation.[11]

**Transfer lesson:** The meaningful shared pattern is localization before classification, with dense/local features preserved. The winning approaches are not evidence that “3D always wins”: the 2025 winner had coordinate and vessel-mask supervision; the 2025 third-place team used ROI crops and spatial feature pooling. Those supervision signals are absent from the knee targets.

### 2024 — Lumbar Spine Degenerative Classification (multi-level MRI)

This is the closest task analogue: MRI, several diagnoses, multiple anatomical levels, and predictions that depend on view and location. Its official objective was sample-weighted log loss, not macro AUC.[27]

**1st — “1st place solution” (NANACHI / Avengers).** The pipeline first estimates disc slice numbers and coordinates, then crops disc-level regions for severity classification. The localizers combine a 3D ConvNeXt slice/instance classifier and coordinate regressor with 2D coordinate models; severity models use adjacent-image groups and attention-based MIL with a bidirectional LSTM. The writeup reports five-fold localization models and median aggregation, plus coordinate perturbations to make crops robust to first-stage errors. The author reports MIL beating simple 2.5D in their final development, but the individual contributions are not all isolated in a matched ablation.[2]

**2nd — “2nd place solution” (Ian Pan, Kevin, Yuji Ariyasu, Bartley).** The team combined distinct axial and sagittal pipelines: slice selection, YOLOX region crops, ConvNeXt classifiers, and five-image MIL. The primary writeup also describes using team OOF predictions to flag high-disagreement training labels (prediction/label difference ≥0.8), with class coefficients, and reports roughly a 1% public/private improvement. This is a late, author-reported experiment on a different task and metric; it is a lead for a cautious knee OOF-label audit, not proof that hard-deleting knee labels helps.[3]

**3rd — “3rd Place Solution” (SonySpine / tkmn / Moyashii).** CenterNet keypoint detectors estimate disc levels and spinal-canal positions; those predictions create sagittal/axial crops. Center and side classifiers use 2D encoders plus attention over slices. The team also expanded training examples by treating slice groups from arbitrary disc levels and, for side targets, canonicalized left/right inputs. They report a 30-model ensemble, but also list 3D CNNs, LSTM sequence models, focal loss, and level-specialized models among unsuccessful variants. The source’s CV/private metric and protocol differ from our knee AUC setting.[18]

**4th — “4th place solution” (SPINE CHART).** This solution combines two keypoint routes, multi-view crops, Transformer/attention pooling, and learned stacking. It reports MLP stacking plus LightGBM/XGBoost experiments; cross-target stacker features did not help, and one residual MLP variant reportedly underperformed a simpler MLP. The writeup’s author says a five-fold multi-view run took at least six hours. The transferable idea is *OOF-tested* stacking of complementary local predictions—not copying the lumbar stacker or its metric objective.[4]

**5th — “5th place solution” (Two People).** Stage 1 trains heatmaps for level-specific points and spreads point labels with Gaussian targets; Stage 2 uses 2.5D CNN–RNN crops and level-wise sequence modeling. The authors report CutMix at p=1.0, two-step training (AUC-oriented pretraining, then frozen-backbone metric fine-tuning), and ensembles across backbones. Several changes are bundled, so causality is unclear; the key useful candidate is a controlled training ablation, not adoption of the entire recipe.[5]

**6th — “6th Place Solution” (NVSpine).** Three complementary pipelines predict coordinates/levels, crop per-disc regions, classify slice sequences, and combine logits using an MLP designed around the competition metric. The authors report four-fold CV and preferring CV to public LB. Some pipeline branches use 2.5D CNN–RNNs, cross-plane level mapping, and target-independent stacking. Their documented failure list includes pseudo-labeling and auxiliary localization for some branches; this is a reminder to test localization and stacking separately.[6]

**Transfer lesson:** Localization and prediction are separable subproblems. The teams had disc-level coordinates; knee labels are study-level, so we cannot directly train their keypoint detectors. The most portable ideas are perturbing localization inputs for robustness, an OOF-based label-quality audit, and testing constrained meta-stacking only after valid OOF predictions exist.

### 2023 — Abdominal Trauma Detection (multi-organ CT)

The challenge used weighted log loss over organ injury and severity labels, not knee AUC.[28]

**1st — “1st Place Solution: Team Oxygen.”** A 3D organ-segmentation model creates crops; each study is sampled into 96 equidistant slices and represented as 32 adjacent-slice triplets. CNN–RNN models predict slice-level injury and segmentation targets; the study label is softened by organ visibility on each slice, and a max over slice predictions forms study output. The authors report four-fold patient GroupKFold and 384×384 inputs. This is a useful example of anatomy masks shaping instance-level supervision, but their CT organ-visibility targets are not lesion coordinates.[7]

**2nd — “2nd Place Solution.”** An organ-presence classifier guides frame sampling, CNN predictions feed a recurrent aggregator, and 3D organ crops feed additional 2D CNN–RNN models. A separately trained LSTM uses organ-conditioned pooling to combine segmentation/presence probabilities and classification features.[13][16]

The author reports a best private loss of 0.35 from MaxViT/crop/CoAtNet-RNN and under-four-hour inference for an unensembled run. The two-stage, organ-conditioned representation is the useful idea; abdominal CT windows and organ masks do not transfer directly to knee MRI.[13][16]

### 2023 — Screening Mammography Breast Cancer Detection (low-transfer contrast)

This competition used pF1, not knee ROC-AUC.[9][19][29]

**1st — “1st place solution” (mr.robot).** A YOLOX breast-ROI detector feeds four-fold, 2048×1024 ConvNeXt-Small classifiers; the author uses several external mammography datasets and reports approximately +0.02 in OOF/private pF1 from the external-data setup.[14][17][19]

The writeup warns that detector validation was small and inconsistent, and its soft-positive-label trick did not clearly outperform label smoothing.[19]

**2nd — “2nd place solution” (cancerdetectman).** The team pretrained a high-resolution single-view ConvNeXt on external data, fine-tuned on competition data, then explored dual/four-view variants. It ultimately favored high resolution and a single-view model; its writeup says experimentation late in the competition focused on one fold, which weakens generality of its comparisons.[20]

**3rd — “3rd Place Solution (Part of data processing and Image-level model)” (H.B.M.F.).** YOLOX crops standardize breast ROIs; image-level CNNs and a multi-view CNN–LSTM combine views, with OOF thresholds and two complementary blends. The team reports that its LSTM-focused blend was more stable across threshold changes than its mean-focused version, while both shared the same displayed public score.[21][32]

**Transfer lesson:** ROI extraction and bag-of-view aggregation are common patterns, but breast-specific views, external mammography data, and thresholded pF1 results are poor direct evidence for knee. Our knee pipeline already has more explicit per-target sequence attention than a simple image-level classifier.

### 2022 — Cervical Spine Fracture Detection (multi-level CT)

**1st — “1st Place Solution” (Qishen Ha).** A 3D vertebra segmentation ensemble trained from 87 mask-labeled studies predicts seven vertebral masks; each crop becomes a per-vertebra sample. A 2.5D classifier stacks five adjacent slices as channels and adds the predicted vertebra mask as a sixth channel; a 2D CNN encodes slices and an LSTM aggregates across 15 slices. A second patient-level model consumes all seven vertebrae to predict overall fracture. The author reports a 7.5-hour submission inference and multiple 384/512-resolution model families; the same writeup says the direct 3D classifier was unsatisfactory.[8]

**Transfer lesson:** Explicit masks can help condition a slice model, but this success had vertebra segmentations and vertebra-specific labels. A knee mask channel is only a candidate if the mask source, train/test path, and laterality semantics are reliable.

## 3. What the current Exp-18 solution already has—and what it does not

This comparison is against the exact local notebook and metadata, not just the separate reusable `src/` library. Exp-18 `57316129` was the current anchor in the live submission query; the source notebook is `notebooks/submission/apex_0957_sota/rsna-knee-apex-0957.ipynb`.

| Mechanism | Exact submitted path status | Assessment |
|---|---|---|
| 2.5D adjacent-slice input and sequence aggregation | **Present** | CoAtNet/Raptor attends over densely sampled three-slice windows; the locally authored ConvNeXt reader uses 16 windows per series and a study Transformer with target-specific attention. |
| High-resolution / multi-resolution inference | **Present** | The Raptor lane includes 384 px and a 224 px sibling; the compact and custom-reader lanes add different pretrained families. |
| Per-finding attention and target routing | **Present** | Raptor uses finding-specific attention over windows; the final rank blend has per-finding weights. The custom reader also has one attention pool per finding. |
| Physical-scale crop and geometric DICOM ordering | **Present in the submitted notebook** | The Raptor lane uses a 140 mm crop, 96-slice corpus, and `k_eval=94`; local `src/dataset.py` also implements physical cropping and geometric slice ordering. |
| Anatomical localization | **Partly present** | The compact R6/R7 inference path can construct knee-centered views from an SKM-TEA anatomy checkpoint when the bundle selects those views. This is not lesion-specific localization. |
| Lesion/compartment ROI supervision or lesion masks | **Not established in the exact submission** | The 2025 aneurysm and 2024 lumbar solutions learn crops from point/mask labels. Knee training labels here are study-level/report-derived, so equivalent lesion targets are absent. |
| OOF-driven label denoising | **Missing / blocked** | The 2024 second-place writeup is a useful lead, but `docs/oof-provenance-audit.md` finds no exact fold-excluded OOF cache for the 0.958 anchor or the exact anchor-plus-specialist stack. |
| Learned meta-stacker on exact OOF streams | **Not established** | The exact apex notebook uses fixed, per-target rank blends. The local ensemble library has ranking/probit utilities, but no verified exact-stack OOF evidence to fit and independently validate a learned stacker. |
| 2.5D/MIL/attention as a general idea | **Not missing** | Re-implementing generic 2.5D, attention pooling, TTA, or another backbone without identifying a distinct error mode would mostly duplicate existing work. |

Two distinctions prevent misleading “we already implemented it” claims:

1. `src/models/slot_head.py` implements six-slot query attention, but the exact 0.958 submission is a separate, self-contained public-model blend; the reusable library is not automatically in that scored path.
2. `src/landmark.py:8-28` selects the slice nearest the median of supplied coordinates. It is not an image-based joint-line detector or a pathology keypoint model. The older `docs/historical_competitions_deep_dive.md` overstates this helper as anatomical localization.

### Corrections to older historical notes

Use this source-verified report instead of treating every claim in `docs/historical_competitions_deep_dive.md` as established fact. The official 2024 second-place writeup describes YOLOX crops, MIL, and OOF-based label filtering; it does **not** support the old note’s specific attribution of a “Wong & Team” query-token solution.[3] The old writeup’s numeric expected AUC gains for transferring prior methods are not supported by the reviewed source writeups. Finally, “3D classifiers systematically lose” is too broad: the 2022 cervical winner reports a 3D classifier was unsatisfactory, while the 2025 aneurysm winner used a 3D segmentation-pretrained classifier with explicit ROI/mask supervision.[8][10]

## 4. Ordered next-experiment backlog

**Priority** (`P0`–`P4`) is the order of work, not a score estimate. Each score-affecting item must pass a *screen* on development folds, then a *confirmation* on untouched held-out folds. The listed experiments are not run results. Keep this backlog separate from the append-only result ledger in `docs/experiments.md`.

### P0 — Build a fold-safe comparison substrate (prerequisite; no score claim)

- **Why first:** The strongest transferable claim from the 2024 denoising and stacking writeups depends on genuine OOF. A submission CSV or a different model’s OOF is not validation of Exp-18.
- **Current status:** `docs/oof-provenance-audit.md` reports that no exact OOF cache for Exp-18 or the exact anchor-plus-ASL stack has been found. The highest-scoring CoAtNet checkpoint is a full-data checkpoint, so its own output cannot become OOF by simply running it on training studies.
- **Work:** Inventory/checkpoint-manifest every candidate branch; recover or regenerate fold-specific predictions only where the corresponding model excluded the study. If exact anchor OOF cannot be reconstructed, create a separate fold-trained candidate baseline and label the experiment as a new lineage rather than “validating Exp-18.” Group by duplicate-report hash and preserve study IDs/fold provenance.
- **Gate:** Exact held-out coverage, one prediction per study per stream, no train-fold overlap, and an auditable checkpoint/split manifest. No ensemble tuning or promotion before this passes.
- **Cost / stop:** High; requires full DICOM inference/training on Kaggle and may require new fold models. If this cannot be made fold-safe, stop instead of substituting public-LB performance or the 58-study gold subset as full-stack OOF.

### P1 — OOF-guided soft correction of report-derived labels

- **Hypothesis:** Some report-derived labels are noisy; fold-held-out image predictions can identify suspicious labels for conservative downweighting or softening. The 2024 second-place team reports using ensemble OOF disagreement to remove samples and improving its own public/private log loss, but that result is unverified for knee and uses a different metric.[3]
- **Change one thing:** Starting from one fixed knee training pipeline and its current report-label table, compare (a) unchanged labels and (b) one predeclared, conservative target-wise OOF-disagreement weighting/softening rule. Begin with downweighting; do not hard-delete every disagreement. Do not simultaneously change the parser, model, loss, and augmentations.
- **Validation:** Train with report-hash-grouped folds; calculate the official knee macro AUC using `src.metrics.compute_competition_metric`. Report per-target results, especially rare findings, plus uncertainty. Keep the 58 human-labeled studies as a separate, explicitly small gold check; never use the same rows both to choose thresholds and to claim confirmation.
- **Gate / stop:** Continue only if improvement recurs across folds and the held-out gold check does not show a material regression. If the change only improves agreement with pseudo-labels or one fold, reject it. No numeric gain is assumed.
- **Dependencies:** P0, report-label provenance, exact fold exclusions, and a threshold policy fixed before confirmation.

### P2 — Test one anatomy-guided knee crop against the current broad view

- **Hypothesis:** A target-specific crop could preserve more local detail than a whole-knee view, analogous to the coordinate/ROI stages in lumbar MRI and mask-guided ROI classifiers in aneurysm detection.[2][10]
- **Change one thing:** Use the existing SKM-TEA knee anatomy anchor to construct one candidate local view for one finding/compartment selected by a preceding error audit. Compare that view with the same model’s broad field; hold the data split, backbone, loss, and blend fixed. Start with a single anatomical region, not a twelve-target detector.
- **Validation:** Check the crop and laterality on a blinded set of representative studies; run matched OOF predictions on the same held-out studies; compare macro AUC and affected-target AUC, crop failure rate, and inference cost.
- **Gate / stop:** Continue only if the ROI is correctly placed across scanners/sides and the isolated OOF change improves the intended target without damaging the macro metric. If no reliable anatomical mask/anchor is available for the chosen region, stop rather than synthesizing pathology coordinates.
- **Dependencies:** P0, exact external-mask provenance/license review, and a documented path proving the mask is available at both train and test time.

### P3 — Add anatomy-mask conditioning to the custom 2.5D reader

- **Hypothesis:** A structure mask or anatomy-conditioned pooling may help the model separate local evidence from background. This is inspired by the sixth mask channel in the cervical winner and the region-masked pooling in the aneurysm winner, but neither result is direct knee evidence.[8][10]
- **Change one thing:** On the custom `KneeNet` reader only, compare the existing three MRI channels against the same input plus one verified anatomy-mask feature (or a single mask-gated pooling operation). Do not alter the pretrained production readers. Keep fold, crop, and training schedule fixed.
- **Validation:** Same held-out folds and targets as the baseline; audit mask quality and left/right semantics first. Report the affected target(s), macro AUC, and missing-mask behavior separately.
- **Gate / stop:** Proceed only if the anatomy mask is reliable and available without using evaluated labels. If the mask is a noisy model-generated pseudo-label or is not train/test-consistent, defer.
- **Dependencies:** P0 and the existing anatomy checkpoint. Additional lesion masks are **not** assumed available.

### P4 — Fit a small OOF meta-blender, not a larger leaderboard stack

- **Hypothesis:** A constrained per-target stacker could outperform the fixed rank weights when complementary streams make different errors. Several lumbar solutions used metric-aware MLP or stacking, but their objective was weighted log loss and their results do not establish a knee AUC lift.[4][6]
- **Change one thing:** First compare a bounded per-target linear rank/probit blend with the existing fixed weights. Only if that passes should a tiny regularized MLP be considered. Fit on development OOF folds and evaluate once on untouched confirmation folds.
- **Validation:** Same subjects and folds for every stream; all predictions must be fold-excluded. Use the official macro AUC and per-target deltas. Keep the stacker constrained to avoid fitting the small 58-study gold subset.
- **Gate / stop:** Reject if gains disappear on confirmation folds, are concentrated in one target/fold, or require using in-sample/full-training predictions. No OOF means no learned stacker experiment.

### Lower priority: CutMix or two-stage loss tuning

The lumbar fifth-place author reports strong CutMix results and two-stage metric fine-tuning, but those experiments bundle multiple changes and target severity log loss.[5] Treat these as later, isolated ablations on the custom training branch—not as an immediate modification to the public pretrained stack. Preserve knee laterality semantics and validate on the same grouped folds.

## 5. What not to copy—and the practical recommendation

- **Do not start a generic full-3D knee classifier because a 2025 solution used 3D.** That winner had dense vessel masks, point labels, a coarse-to-fine ROI, and segmentation-pretrained features; the current knee task does not have equivalent lesion supervision. The cervical winner reports its own 3D classifier failed.[8][10]
- **Do not add another generic attention model or more TTA first.** Exp-18 already has target-specific window attention, 2.5D slices, 384/224 views, compact fold models, anatomical mirror/flip TTA, and per-target rank routing in its exact submission notebook.
- **Do not copy the other competitions’ numeric scores, thresholds, class weights, or pixel crops into knee.**
- Lumbar spine and abdominal trauma use weighted log-loss objectives; cervical spine uses weighted multilabel log loss.[27][28][30]
- The aneurysm challenge uses weighted multilabel AUC.[26][31]
- Mammography uses pF1, which is not numerically interchangeable with knee macro AUC.[9][19][29]
- External-data and annotation lineages differ, so review eligibility and licenses before reuse.[1][10][16]
- **Do not call the current series-selection inventory or local `landmark.py` a score result.** The inventory is metadata-only; `landmark.py` picks a median-coordinate slice; neither established a knee AUC gain.

**Recommendation:** The next score-affecting experiment should be **P1, but only after P0 produces valid OOF**. It is the highest-value unclosed idea because report-derived labels dominate the training set, while the current exact scorer has already exhausted the obvious inference-side recipe. After that, test one anatomy-guided crop (P2), then a mask-conditioned reader (P3), then a constrained OOF stacker (P4). None is credible as a new final-day submission in the remaining competition window; this document is a hypothesis backlog for a properly validated training cycle.

## 6. Evidence and artifact boundaries

- The source set is selective; several placements and all public notebook versions were not exhaustively examined.
- Official result pages confirm the 2025 aneurysm, 2024 lumbar, and 2023 abdominal placements.[22][23][24]
- RSNA’s 2022 cervical result page confirms that competition’s placements.[25] These outcome pages do not independently verify individual ablations.
- The 2025 aneurysm metric is weighted multilabel AUC.[31]
- Lumbar spine and abdominal trauma use weighted log-loss objectives.[27][28]
- Cervical spine uses weighted multilabel log loss.[30] Mammography uses pF1.[19][29] Scores are not comparable across these challenges.
- No public notebook code was executed. The current local notebook was statically inspected only. No new predictions, CV, training, submission, or final-slot action occurred.
- Promising hypotheses belong here; measured results belong in `docs/experiments.md` after they actually run.

## Sources

[1] https://www.kaggle.com/competitions/rsna-intracranial-aneurysm-detection/writeups/2nd-place-solution
[2] https://www.kaggle.com/competitions/rsna-2024-lumbar-spine-degenerative-classification/writeups/avengers-1st-place-solution
[3] https://www.kaggle.com/competitions/rsna-2024-lumbar-spine-degenerative-classification/writeups/ianpan-kevin-yuji-bartley-2nd-place-solution
[4] https://www.kaggle.com/competitions/rsna-2024-lumbar-spine-degenerative-classification/writeups/spine-chart-4th-place-solution
[5] https://www.kaggle.com/competitions/rsna-2024-lumbar-spine-degenerative-classification/writeups/two-people-5th-place-solution
[6] https://www.kaggle.com/competitions/rsna-2024-lumbar-spine-degenerative-classification/writeups/nvspine-6th-place-solution
[7] https://www.kaggle.com/competitions/rsna-2023-abdominal-trauma-detection/writeups/team-oxygen-1st-place-solution-team-oxygen
[8] https://www.kaggle.com/competitions/rsna-2022-cervical-spine-fracture-detection/writeups/qishen-ha-1st-place-solution
[9] https://www.kaggle.com/c/rsna-breast-cancer-detection
[10] https://www.kaggle.com/competitions/rsna-intracranial-aneurysm-detection/writeups/1st-place-solution
[11] https://www.kaggle.com/competitions/rsna-intracranial-aneurysm-detection/writeups/3rd-place-solution
[12] https://github.com/uchiyama33/rsna2025_1st_place/blob/main/solution_v2.md
[13] https://github.com/TheoViel/kaggle_rsna_abdominal_trauma
[14] https://www.kaggle.com/competitions/rsna-breast-cancer-detection/discussion/392449
[15] https://raw.githubusercontent.com/uchiyama33/rsna2025_1st_place/main/solution_v2.md
[16] https://raw.githubusercontent.com/TheoViel/kaggle_rsna_abdominal_trauma/cleaning/README.md
[17] https://github.com/dangnh0611/kaggle_rsna_breast_cancer
[18] https://www.kaggle.com/competitions/rsna-2024-lumbar-spine-degenerative-classification/writeups/sonyspine-s-tkmn-moyashii-3rd-place-solution
[19] https://www.kaggle.com/competitions/rsna-breast-cancer-detection/writeups/mr-robot-1st-place-solution
[20] https://www.kaggle.com/competitions/rsna-breast-cancer-detection/writeups/cancerdetectman-2nd-place-solution
[21] https://www.kaggle.com/competitions/rsna-breast-cancer-detection/discussion/391779
[22] https://www.rsna.org/news/2025/november/2025-ai-challenge-winners
[23] https://www.rsna.org/news/2024/november/2024-ai-challenge-winners
[24] https://www.rsna.org/news/2023/november/rsna-ai-challenge-abdominal-trauma-detection-results
[25] https://www.rsna.org/news/2022/november/ai-challenge-2022-winners
[26] https://www.kaggle.com/competitions/rsna-intracranial-aneurysm-detection/overview/evaluation
[27] https://www.kaggle.com/competitions/rsna-2024-lumbar-spine-degenerative-classification/overview/evaluation
[28] https://www.kaggle.com/competitions/rsna-2023-abdominal-trauma-detection/overview/evaluation
[29] https://www.kaggle.com/competitions/rsna-breast-cancer-detection/overview/evaluation
[30] https://www.kaggle.com/competitions/rsna-2022-cervical-spine-fracture-detection/overview/evaluation
[31] https://www.kaggle.com/competitions/rsna-intracranial-aneurysm-detection
[32] https://www.kaggle.com/competitions/rsna-breast-cancer-detection/writeups/h-b-m-f-3rd-place-solution-part-of-data-processing
[33] https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/overview/timeline — RSNA Knee official timeline
