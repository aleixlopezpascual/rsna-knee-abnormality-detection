# Deep Dive: Kaggle Discussion #745861 — “Tricks for the competition”

- **Reviewed:** 2026-10-22 (Kaggle topic API refresh at 10:20 UTC)
- **Thread:** [745861](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/745861)
- **Outcome:** Research and claim audit only. No model, Kaggle kernel, submission, or final-selection change was made.

## Executive conclusion

The thread contains several useful **hypotheses**, especially separating explicit report evidence from “not mentioned,” using target-specific confidence/abstention, and asking an image model to review uncertain cases. It is not a validated solution write-up: the original poster authored 31 of the 52 comments returned by Kaggle, several attachments are ChatGPT-generated suggestions, and most reported scores lack a reproducible notebook, split manifest, or independent confirmation.[41]

The most important correction is numerical: the post calls a **0.938** gold-58 blend “local CV,” but its attached diagram labels the per-target-tuned result **“in-sample / optimistic.”** Another screenshot explicitly says the same gold studies were used to select and evaluate label-state combinations. Neither number is a clean held-out result.[41][44][48]

A more serious factual correction: the author’s example claims that a gold study with report text “No knee effusion is present” has `Effusion=1`. For `StudyInstanceUID=1.2.826.0.1.3680043.8.498.12606657226568558340797193167488111973`, the matching row in our official `input/train.csv` has **`Effusion=0`**. That example therefore does not demonstrate report/gold disagreement as stated; it appears to be a mistaken label or copy.[41] (Verified locally by exact UID lookup.)

**Practical takeaway:** keep the thread’s report-state/abstention and image-assisted review ideas in the future experiment backlog, but do not treat them as established score gains. For our current stack, a fold-safe comparison substrate is still the prerequisite; the exact Exp-18 anchor-plus-specialist OOF stream is not available in the local provenance audit.[docs/oof-provenance-audit.md:9-15,60-70]

## Scope and evidence rules

- The user-provided capture showed 40 comments. At the refresh for this report, Kaggle’s topic API returned **52 comment records**; 31 were by the original poster, `hengck23`. Thread counts are dynamic, so the number here is an as-of snapshot, not a permanent property.[41]
- Kaggle’s page extractor was rate-limited, so I read the full topic/comment payload through Kaggle’s API/CLI.[41] I visually inspected the substantive image attachments; `Selection_5207.png` returned 404 and is not used as evidence.[41] The original post’s supplied capture is retained as context, and the audit below checks its claims against accessible comments, attachments, local competition files, papers, dataset cards, and host rulings.[41]
- **Official host/rule statement** means a direct organizer answer. **Paper/dataset/repository statement** means the source’s own description or result, not independent reproduction. **Participant-reported** means a Kaggle comment or personal result. **Hypothesis** means a proposed method without measured confirmation.
- The citations support the source statements; they do not turn author-reported LB or 58-study results into independent evidence.[41][64]
- In particular, “AUC” without a split, checkpoint, and holdout protocol is not called CV here.[44][48]

## Claim-by-claim audit

| Thread claim or idea | What the evidence actually establishes | Verdict for our use |
|---|---|---|
| “Show ChatGPT the report and image; ask it to label the ROI.” | This is the original poster’s proposal. The thread does not report ROI-localization accuracy, a localization benchmark, or an ablation comparing LLM-selected crops with a fixed crop.[41] | **Promising idea, not a result.** Treat report-plus-image ROI annotations as a teacher/triage hypothesis; do not assume the generated boxes are correct. |
| Use the 4.3k report-bearing studies as weakly labelled MRI training data. | Our local `input/train.csv` contains 4,407 studies; 58 have all 12 official labels and 4,349 have reports without that complete gold target row. The competition host explicitly confirmed that LLM APIs may read reports to generate labels, subject to the competition rules, service accessibility/cost conditions, and inference-input restrictions.[35] | **Confirmed training opportunity.** Test-report absence means report-derived features must be offline supervision/teacher signals; the submitted model still needs to run on allowed MRI inputs.[AGENTS.md:90-93] |
| Claude Opus 4.5 labels score about 0.892 on gold-58; a Qwen text student about 0.889; an image-only MaViT student about 0.908. | These values appear in the poster’s attached diagram, not in an independently reproduced run. The diagram says all models were evaluated on the same gold-58 set; the source code, folds, prompts, exact checkpoints, and prediction files are not supplied in the thread itself.[41] | **Author-reported screenshot results.** Useful for forming hypotheses, not proof of label quality or CV performance. |
| Fusing those predictions gives 0.928 fixed 50/50 and 0.938 per-label alpha. | The diagram reports a 0.928 fixed blend and 0.938 tuned blend, but explicitly labels the latter **“in-sample / optimistic.”** Thus the comment’s “local CV on gold58 = 0.938” wording is inconsistent with its own attachment; the 0.938 is post-selection on the same small gold set, not independent CV.[44] | **Reject as a validated CV result.** Do not use 0.938 as evidence for promotion or as a target for a new blend. |
| A many-model/gold-58 hyperparameter search reaches about 0.938. | The spreadsheet screenshot labels the training corpus “single fold = 4.3k”; it contains several model/resolution/blend trials and a top displayed combination around 0.93799. It does not provide an untouched confirmation set; the labels and combinations are selected against gold-58.[45] | **Development search only.** The table itself does not establish generalization. |
| Evidence-state filtering improves macro AUC from 0.8919 to 0.9159. | The screenshot says it requires at least 20 gold samples per selected combination, but then explicitly states that the same 58 gold studies were used to select and evaluate the combinations, which are different subsets by class. It calls 0.9159 **in-sample, post-selection AUC** and warns that it does not establish higher AUC on unseen studies.[48] | **Not a validated lift.** This is a useful prompt for a controlled experiment, not a score claim. |
| Multi-LLM agreement filtering (keep probabilities >0.9 or <0.1) removed about 25% of labels and improved CV/LB. | This is a participant’s short self-report. The comment gives no fold assignment, exact AUCs, target-by-target results, sample denominator, model versions, or code.[41] | **Unverified hypothesis.** Test target-wise and against the exact MRI baseline; don’t discard entire studies because one target is uncertain. |
| A 58-study model with roughly 10 parameters is “the only” method avoiding overfit/shakeup; prior is the key. | No controlled comparison or learning curve supports the “only method” or 10-parameter threshold. The accompanying notebook screenshot is a Raptor + ConvNeXt-S6 blend with pretrained datasets/checkpoints and a displayed best score, not a demonstrated model trained only on 58 gold studies.[41][47] The hyperparameter screenshot also describes image/text students trained using the 4.3k weak-label corpus.[45] | **Generalization not established.** Frozen pretrained features plus a small head is a reasonable low-variance experiment; parameter count alone is not a sample-size guarantee. |
| The displayed gold example proves the reports can contradict gold labels. | For the exact UID in the post, the official local `input/train.csv` row contains the report sentence “No knee effusion is present” and the official `Effusion` value is **0**, not 1. The thread’s quoted `Effusion=1` is contradicted by the local official row.[41] | **Contradicted example.** Do not cite it as evidence of annotation/report disagreement. The broader mismatch may exist, but this example does not establish it. |
| “3DReasonKnee” provides boxes/segmentations for most of the 12 competition abnormalities. | The paper describes 494k question–image–box–reasoning–severity quintuples from 7,970 OAI DESS MRI scans. Boxes are generated from nnU-Net outputs trained using 100 manually segmented scans; questions and grades follow MOAKS, including cartilage, bone marrow lesions, osteophytes, meniscal damage, ligament/tendon, effusion-synovitis, and periarticular features.[55] The HF card describes OAI DICOM access, MOAKS labels, masks, and a CC BY 4.0 dataset license.[72] | **Partly related, not a direct 12-class label set.** The boxes localize MOAKS anatomical subregions; a one-to-one mapping to RSNA ACL/MCL/OA/effusion/etc. criteria is not established. This may supply an anatomy prior, not gold RSNA targets. |
| The OAI tissue-mask dataset supplies labels for the 12 RSNA classes. | The Kaggle dataset card describes model-derived masks for femur, tibia, patella, cartilage, and meniscus from OAI DESS-we MRI, covering 46,824 scans/4,795 participants. Its Dice table concerns tissue segmentation—not 12-class injury diagnosis.[63] | **Incorrect as a direct label claim.** Potential anatomy/localization supervision only; it does not directly label ACL tears, MCL tears, fractures, Baker’s cysts, or all competition classes. The masks are inferred outputs, not all manual pathology annotations. |
| KJD-25K / KneeCoT is a useful external training source. | The MICCAI KJD-25K page links its dataset to `YiHui0124/KneeCoT`.[57] The competition host explicitly ruled that **KneeCoT was not allowed** because its required institutional data agreement could make access uneven.[42] | **Do not use KneeCoT for a competition submission.** KJD-25K’s paper can inform observation/design, but the host’s dataset-specific ruling controls use of that linked dataset. This ruling does not automatically cover separate OAI mask resources. |
| OrthoFoundation is a relevant MRI prior with strong ACL/MCL results. | The OrthoFoundation paper reports a DINOv3-L, self-supervised model evaluated after downstream fine-tuning, with MRI injury tasks; its original paper describes about 1.25M knee images and includes an internal PUTH cohort plus OAI/fastMRI.[70] The expanded MSK-5M README separately describes a 5M-image checkpoint (about 4M knee and 1M cross-joint images) and repeats downstream metrics, including 98.49 ACL AUROC and 94.39 ACL AUROC at 12.5% labels.[71] | **Relevant prior, but not directly comparable.** These are different datasets, label definitions, and evaluation protocols—not RSNA 12-class macro AUC. The 5M checkpoint and the original paper’s 1.25M corpus should not be conflated. The paper’s internal cohort also raises a source-eligibility question. |
| The cited OAI TKR paper shows OAI MRI is useful for this competition’s labels. | The paper predicts **total knee replacement within five years**, using OAI DESS MRI plus clinical/demographic information; it reports AUC 0.834 overall and 0.943 for the no-radiographic-OA subgroup.[56] | **Narrowly confirms prognostic signal, not transfer performance.** TKR prognosis is not the same endpoint as the competition’s 12 MRI abnormality labels. |
| “JEV” is a knee-abnormality labeler. | The linked Jev paper evaluates Jev as a report-factuality judge: it scores whether statements are supported, contradicted, or unaddressed relative to physician-written reports. It is not, by itself, a validated 12-target knee diagnostic labeler.[65] | **Ambiguous use in the thread.** Without the exact prompt/model pipeline, “Qwen JEV” cannot be treated as a well-defined target source. |
| A rubric-first “controlled CoT” or “abstain/unknown” labeler improves MRI training. | The attached prompt screenshots propose atomic report questions, explicit evidence categories (positive/negative/below-threshold/incomplete/conflicting/unmentioned), and strong/weak/abstain trainability. Another assistant screenshot proposes class-specific reliability filtering.[46][49][62] These are prompt/workflow suggestions, not measured model results. | **Good experimental design idea, no demonstrated gain.** Preserve unknown/conflict as a masked/weighted target state rather than converting silence to negative; test it per target. |
| Expensive LLM calls are worth it. | The thread includes a personal account of using a large share of one user’s credit quota; the screenshot is an account-balance snapshot, not a reproducible per-study or per-label cost benchmark.[41] | **Cost is pipeline-specific.** Budget by a measured pilot, not the anecdotal credit balance. Consider uncertainty-triggered review rather than labeling every case with the most expensive model. |

## The “effusion=1” example: local verification

I filtered [`input/train.csv`](../input/train.csv) for the exact UID in the thread. There is one matching row; the report says “No knee effusion is present” and the official label is `Effusion=0`, contradicting the thread’s `Effusion=1` claim.[41] The 58 gold rows have only 9 MCL positives, 11 Lateral OA, 12 Baker’s cyst, and 15 Medial OA positives; the remaining target counts range from 18 to 35. These small per-target counts make post-hoc prompt/blend selection on the same 58 particularly vulnerable to noise and selection bias.[44][48]

## What transfers to our exact solution—and what does not

| Thread mechanism | Status in this repository / scored path | Practical interpretation |
|---|---|---|
| MRI-only model at test; report-based labels at training | **Partly present** | The repository has a deterministic multilingual report extractor with continuous scores (`src/report_continuous_extractor.py`); it is not a frontier LLM. The report-transfer idea is already represented in our tooling, but that does not prove the exact Exp-18 checkpoint was trained using every local parser output. Test reports are absent, so report inputs cannot be used directly at inference.[AGENTS.md:90-93] |
| 2.5D windows, multiple views, pretrained MRI models, per-target blending | **Already present** | The exact Exp-18 path already has dense 2.5D windows, 384/224 image streams, pretrained readers, TTA, and target-specific rank blending.[docs/rsna_prior_solution_transfer_review.md:83-98] Rebuilding a generic multi-view student would duplicate existing work. |
| Anatomy/ROI localization | **Partly present** | The exact stack uses physical crops and has an OrthoFoundation-related source attached; the prior audit marks localization as partial, not lesion-specific. The 0.958 kernel metadata attaches `heliosli/rsna-orthofoundation-fold0-epoch7-20261004`, but that does not prove it uses the newly linked expanded `OrthoFoundation-MSK-5M` DINOv3-L checkpoint.[docs/oof-provenance-audit.md:17-31] |
| Report evidence state + abstention as a supervised loss mask | **Not established in the exact scored path** | This is the clearest incremental thread idea: keep a per-study/per-target state and confidence; mask or downweight only the unreliable target rather than discarding the whole study. The thread’s 0.9159 result is not independent evidence for it.[48][49] |
| OOF-driven correction of weak labels | **Blocked for the exact current stack** | The local audit found no exact OOF cache for Exp-18 or the anchor-plus-specialist stack. One separate reproducible repository reports that its own OOF label repair worsened gold macro AUC from 0.8955 to 0.8891; that is a cautionary result from another model, not proof about our stack.[docs/oof-provenance-audit.md:9-15,60-70][64] |
| OAI tissue masks / 3DReasonKnee boxes | **Potential anatomy prior; access and domain gaps** | OAI uses DESS sequences and MOAKS; our competition data are multi-sequence clinical MRI. The masks/boxes may help anatomy localization or pretraining but are not direct 12-class targets and cannot be assumed to transfer.[55][63] |
| Expanded OrthoFoundation model | **Related model family exists; exact checkpoint is different** | The current pipeline already attaches an OrthoFoundation-related fold checkpoint, but the separate 5M DINOv3-L checkpoint would be a new model lineage. The repository reports an internal PUTH data cohort as part of pretraining; external-data eligibility and license provenance must be reviewed before use.[70][71][35] |

The external-data point is separate from model performance. The competition host’s general guidance says readily accessible non-commercial research datasets are not excluded solely for being non-commercial, but provider-specific terms remain the team’s responsibility.[35]

The OAI portal says access requires an NDA account and describes use for human-subject research for scientific and educational purposes.[58]

In the sources reviewed here, I found no dataset-specific ruling authorizing the exact 3DReasonKnee or 5M OrthoFoundation checkpoint for a prize-eligible submission.[35][42]

Do not treat a general rule or the HF/Kaggle dataset card as a waiver of the upstream OAI/clinical-cohort terms.[72][docs/final-submission-selection.md:30-34]

## Ordered experiment ideas (hypotheses, not results)

Priority is **work order**, not estimated AUC gain. Every score-affecting idea requires an exact baseline, the same held-out studies/folds, and the official competition metric. No numeric improvement is assumed.

### P0 — Re-establish a fold-safe substrate for the exact candidate

- **Why first:** The thread’s tuned gold-58 blends and label filters cannot validate our production ensemble. The existing OOF audit says the exact Exp-18 streams and anchor-plus-specialist OOF are missing.[docs/oof-provenance-audit.md:9-15,60-70]
- **Work:** Checkpoint/split-manifest inventory; recover only predictions whose checkpoint excluded each study. If impossible, create a distinct fold-trained research lineage instead of calling it Exp-18 validation.[unverified]
- **Gate:** One verified fold-excluded prediction per study per stream, exact ID alignment, group isolation, and target provenance. No blend tuning before this passes.[unverified]
- **Status:** Prerequisite; not executed in this review.

### P1 — Evidence-state, class-specific weak-label masking

- **Hypothesis:** Explicit positive, explicit negative, below-threshold, incomplete, conflicting, and unmentioned states may be more reliable supervision than forcing every report into 0/1. The thread’s prompt sketches suggest this structure but provide no independent result.[49][62][unverified]
- **Experiment:** Keep the current parser/model/split fixed. Compare (a) current continuous report targets, (b) a predeclared target-wise mask/weight for conflict, insufficient evidence, and silence, and (c) a conservative confidence-weighted loss. Keep other targets from the same study; do not drop the whole study because one class is uncertain.[unverified]
- **Validation:** Report macro AUC and all 12 target AUCs on fold-excluded MRI predictions against the gold labels. Use the 58 gold studies only as a small development check; because they have been repeatedly inspected, they are not an untouched confirmation set.[unverified]
- **Stop:** Reject changes that only improve agreement with weak labels, rely on a post-selected gold subset, or regress the full macro metric.[unverified]

### P2 — OOF-guided, target-level label correction (only after P0)

- **Hypothesis:** OOF model/report disagreement can help identify selected weak labels for review. However, the thread’s 0.938/0.9159 figures are post-selection, and another public repository reports its own label repair worsened gold macro AUC.[44][48][64][unverified]
- **Experiment:** Compare unchanged labels with one predeclared conservative softening/downweighting rule, target by target. Consider image-based teacher correction only where the teacher prediction is genuinely fold-excluded. Do not use the same 58 gold studies both to fit correction thresholds and to claim confirmation.[unverified]
- **Stop:** If exact OOF provenance is unavailable, or if gains occur only on the gold set used for tuning, do not promote.[unverified]

### P3 — Selective image-aware annotation / active review

- **Hypothesis:** A vision-language teacher or human reviewer may resolve image/report uncertainty for a small set of high-value cases. The thread reports a small improvement from hand-crafted slice features but gives no metric, split, or reproducible pipeline.[41][unverified]
- **Experiment:** On a fixed pilot, select uncertain cases using the existing MRI model’s fold-excluded predictions; compare report-only labels with image+report review, preserving laterality and the official rubric. Record annotation cost per reviewed target and the exact slices shown.[unverified]
- **Stop:** Do not scale based on a single AUC computed after tuning on the same gold-58 examples. Keep inference MRI-only and verify API/access rules.[35][unverified]

### P4 — One anatomy-mask or crop ablation from OAI resources

- **Hypothesis:** The OAI masks and 3DReasonKnee region annotations could support an anatomical prior, especially for cartilage/meniscus localization, but are not direct RSNA targets.[55][63][72][unverified]
- **Constraint:** The DESS-to-multisequence domain shift and external-data terms are additional blockers.[58][35]
- **Experiment:** After provenance/legal review, test one target-specific anatomy mask/crop in the custom reader, compared with the existing broad view on matched folds. Measure mask failure, laterality errors, inference cost, and target/macro AUC.[unverified]
- **Stop:** Do not use KneeCoT (host explicitly disallowed it), and do not treat OAI access or a derived dataset license as automatic competition approval.[42][72][unverified]

### P5 — Expanded OrthoFoundation checkpoint comparison

- **Hypothesis:** The 5M-image DINOv3-L checkpoint may add complementary features, but the thread’s reported downstream metrics are from different tasks/labels and are not directly comparable with RSNA macro AUC. The project already contains an OrthoFoundation-related stream, so this must test the incremental expanded checkpoint rather than re-adding the same lineage.[70][71][docs/oof-provenance-audit.md:19-31][unverified]
- **Experiment:** Verify checkpoint identity, exact pretraining lineage, license/data access, and existing model correlation. Then run a single controlled encoder/stream replacement or small blend ablation on identical folds.[unverified]
- **Stop:** Do not proceed until private-cohort/OAI lineage and award eligibility are reviewed; do not promote from the paper’s ACL AUROC alone.[35][70][71][unverified]

### Not recommended from this thread

- Do not interpret **0.938** as an independent CV score; the screenshot itself marks it in-sample/optimistic.[44]
- Do not treat `Effusion=1` in the quoted example as a real gold/report contradiction; the official local CSV gives 0.[41]
- Do not train a fresh high-capacity vision model on only 58 gold cases because of the “10 parameters” assertion; the thread gives no such controlled result.[41][45][47]
- Do not hard-delete 25% of studies based solely on one participant’s agreement threshold; the post provides no reproducible protocol or per-class result.[41]
- Do not add out-of-domain body-part MRIs as assumed negatives without a controlled distribution-matched ablation; the thread gives no supporting competition result.[41]
- Do not use the KJD-25K-linked KneeCoT data for competition training; the host’s specific ruling is explicit.[42][57]

## Reconciliation with an older project gap note

`docs/competitive_advantages_and_gap_analysis.md:48-59` calls the report-label frontier “the #1 decisive factor,” attributes proprietary LLM pipelines to top teams, and assigns a +0.006–0.010 AUC lift. This thread and the primary sources reviewed here do not substantiate those broad attributions or numeric gains. They support that report-derived weak labels are an important training route and that report-reading LLM APIs are permitted under conditions; they do not show a causal AUC gain for our exact stack. Community reports are mixed, and one separate open repository reports that its own OOF label repair reduced gold macro AUC.[35][41][64] Treat the older numerical gains and undocumented “top-team” recipes as hypotheses until backed by a source or a controlled local ablation.

## Bottom line for this competition

This thread is useful as an **idea mine**, especially for label provenance, abstention, target-specific reliability, and image-guided review.[41][49][62]

It does **not** show that LLM relabeling or 58-shot low-parameter training is the single winning trick.[41][44][47]

The strongest reported number in the thread is not a clean held-out score.[44][48]

The gold/report “contradiction” is false against our official CSV, while the external datasets/checkpoints carry sequence and eligibility caveats.[41][55][63]

The best next controlled research item is **P1 after P0**: add auditable evidence-state/abstention handling to weak labels and evaluate on genuinely fold-excluded predictions.[unverified] Do not change the final 0.958 entry or launch a late leaderboard experiment based on this thread. This review itself changed no model, no submission, and no Kaggle selection.

## Sources

[35] https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/733965
[41] https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/745861 — Kaggle discussion 745861 — Tricks for the competition
[42] https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/734109
[44] https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F113660%2Fba0c65b0c737533e01fc3ca07a3b4564%2FSelection_5146.png?generation=1791619623246016&alt=media
[45] https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F113660%2Fad1826c4b56d443ee8567f96539704bd%2FSelection_5203.png?generation=1791803327548434&alt=media
[46] https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F113660%2Feb61b0c47696d4276d7406e1f2bcb9a6%2FSelection_5202.png?generation=1791802007559291&alt=media
[47] https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F113660%2F79547bd0a7f30b4204c807e56cacee10%2FSelection_5193.png?generation=1791761088508250&alt=media
[48] https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F113660%2Fbb76c4baa676020353a676a042629480%2FSelection_5205.png?generation=1791829900480092&alt=media
[49] https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F113660%2Ffbd75b3df486c407e3c7bef1cefafbb1%2FSelection_5209.png?generation=1791830693971330&alt=media
[55] https://arxiv.org/html/2510.20967v1
[56] https://arxiv.org/html/2002.10591v1
[57] https://papers.miccai.org/miccai-2026/0534-Paper0576.html
[58] https://nda.nih.gov/oai/query-download
[62] https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F113660%2F3b294e4b65b93716f32f1d3b65a3d1a03bce%2FSelection_5213.png?generation=1791836796315729&alt=media
[63] https://www.kaggle.com/datasets/kgaooo/oai-tissue-segmentations
[64] https://github.com/NTejas-1/RSNA-Knee-Abnormality-Detection
[65] https://arxiv.org/html/2609.27607v1
[70] https://arxiv.org/pdf/2601.18250v1 — OrthoFoundation arXiv paper
[71] https://raw.githubusercontent.com/ytrsk/OrthoFoundation-MSK-5M/main/README.md — OrthoFoundation-MSK-5M raw README
[72] https://huggingface.co/datasets/rajpurkarlab/3DReasonKnee/raw/main/README.md — 3DReasonKnee Hugging Face data card README
