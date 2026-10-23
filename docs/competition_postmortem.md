# RSNA Knee Abnormality Detection: Competition Post-mortem

**Date:** October 23, 2026

## 1. Executive Diagnosis

The repository produced a strong engineering pipeline and achieved a solid public score but outside the leading group, featuring a sophisticated 2.5D multi-resolution and attention-routed architecture. However, the effort was fundamentally hampered by a severe late start and a critical gap in validation discipline: the lack of exact-candidate out-of-fold (OOF) validation against radiologist ground truth. Without exact OOF, the team was forced to rely on public Leaderboard (LB) probing and validation against weak report-derived labels, introducing significant selection risk and limiting confidence in the final private standing.

## 2. Evidence & Status Snapshot

*   **Retrieval Date:** Kaggle submission API queried on 2026-10-23 11:37 CEST.
*   **Submission Scores (Public/Private):**
    *   Exp-18 (Ref 57316129): **0.958** / **0.942**
    *   Exp-19 (Ref 57365961): **0.957** / **0.941**
    *   Exp-20 (Ref 57410300): **0.957** / **0.941**
    *   Exp-22 (Ref 57478743): **0.958** / **0.942**
    *(Note: Scores are displayed to three decimals. The 0.016 Exp-18 public/private difference is a warning signal, not a diagnosis of public-LB overfit or distribution shift; the private leaderboard remains preliminary and scores are rounded to three decimals. Exp-22 was explicitly a public-LB-only authorized experiment and tied Exp-18 at displayed precision; it does not constitute evidence of a performance gain.)*
*   **Leaderboard Snapshot:** Downloaded public LB snapshot timestamped 2026-10-23T09:37:21 (6,425 rows). Our team (Aleix López) ranked **1,033** with a public score of **0.958**. The highest public score was **0.970**.
*   **Current Competition Status:** The Kaggle competition page indicates the competition has ended, but the private leaderboard is preliminary and pending verification. The public rank is the dated public snapshot, while the final private rank and prize result remain unfinalized because the private leaderboard is preliminary; separately, earlier project documentation said the final-selection UI was not changed.
*   **Timeline:**
    *   Competition Start: 2026-07-30
    *   First Repository Commit: 2026-09-30 (62 days late, leaving only 22 days).
    *   Final Submission Deadline: 2026-10-22

## 3. Root-Cause Analysis

*(Ranked by evidence)*

1.  **Late Start (Verified):** The first commit occurred 62 days into the competition, consuming the vast majority of the timeline and severely constraining the runway for robust experimentation, OOF generation, and iterative improvement.
2.  **Validation Mismatch and Weak Labels (Verified):**
    *   The OOF audit (`docs/oof-provenance-audit.md`) confirms no verified fold-excluded predictions exist for the exact Exp-18 stack or the exact anchor-plus-specialist stack.
    *   While `src/dataset.py:160-204` correctly implements report-hash-grouped multilabel folds, `src/train.py:253-275` demonstrates that the default training path utilizes report-derived labels.
    *   *Crucial distinction:* Cross-validation against weak, report-derived labels is not equivalent to validation against the true radiologist ground truth. Weak-label CV and the absence of exact-stack OOF predictions left actual generalization to the hold-out target distribution uncertain.
3.  **Public-LB Selection Risk (Verified):** Lacking strict exact-candidate OOF predictions, the experiment history indicates the project relied on public LB scores to evaluate complex, multi-stage ensembles. This represents a serious validation and selection risk supported by the experiment history, though it is not a proven cause of the private score gap.
4.  **Documentation / evidence overclaiming (Verified):**
    *   Project artifacts establish unsupported claims in documentation, but because complete agent invocation logs are unavailable, these claims cannot be attributed to agents as a verified fact. The research review (`docs/rsna_prior_solution_transfer_review.md`) explicitly corrected unsupported numeric gain claims and misattributions present in older documentation (e.g., `docs/historical_competitions_deep_dive.md`).
    *   *Failure Mode:* These unsubstantiated claims represent an evidence-discipline failure that risks misdirecting effort; however, direct score causality is unproven.
5.  **Ensemble Complexity, Diversity, and Lineage (Hypothesis):** Combining multiple pre-trained model lineages without exact OOF validation made it difficult to verify if the added complexity provided genuine orthogonal diversity or merely diluted the strongest anchors.

## 4. What Worked (To Be Reused)

The exact Exp-18 path established a highly capable baseline architecture. The following engineering patterns were successfully implemented and should be retained for future medical imaging tasks:

*   **2.5D Windowing:** Aggregating adjacent slices to capture local volumetric context without the memory penalty of pure 3D CNNs.
*   **Multi-Resolution / High-Resolution Pipelines:** Utilizing 384px and 224px input sizes for different ensemble branches.
*   **Target-Specific Attention / Routing:** Allowing the model to focus on the anatomical views most relevant to specific pathologies.
*   **Physical-Scale Cropping:** Standardizing the field-of-view in millimeters to ensure consistent zoom levels across varying scanner resolutions.
*   **Geometric Slice Ordering:** Robustly sorting DICOM slices using 3D physical coordinates rather than unreliable file names.
*   **Anatomical Mirror TTA:** Properly handling laterality to prevent flipping augmentations from swapping medial/lateral semantics.

*(Note: Generic `SlotHead` and `landmark.py` helpers were present in the repository but were not necessarily part of the exact scored Exp-18 path.)*

## 5. AI-Agent Retrospective

*(Caveat: Exact agent invocation and time logs are unavailable. This evaluates artifacts and processes, not imagined agent actions.)*

Invocation and time logs are unavailable; repository artifacts demonstrate the risks of unverified research and documentation claims, but they do not establish which agent or person produced them or that collaboration was unstructured. Moving forward, the workflow must enforce strict discipline.

### Improved Split-Role Workflow
Agents should be designated explicit, constrained roles to prevent scope creep and ensure rigorous verification across workstreams:
1.  **Solution Researcher:** Mines exact top-team code and writeups from this and analogous completed competitions; inspects underlying code rather than inferring capabilities from notebook titles; reports source citations, validation/score evidence class (`[VERIFIED]`, `[REPORTED]`, `[HYPOTHESIS]`), dependencies, and transfer limits.
2.  **Data & Label Auditor:** Verifies split integrity, leakage prevention, and label provenance (distinguishing radiologist ground truth from weak report-derived labels).
3.  **Data / Feature Exploration Agent:** Performs automated EDA and proposes feature, subgroup, or error-analysis hypotheses from observed data and valid OOF predictions; outputs hypotheses, not claimed gains.
4.  **OOF Auditor:** Maintains the definitive manifest and provenance of exact fold-excluded predictions.
5.  **Experiment Analyst:** Converts hypotheses into a ranked one-factor backlog with cost, falsification/acceptance gates, and an untouched confirmation plan.
6.  **Gemini Implementer:** Handles scoped code creation and refactoring only after an experiment is selected from the backlog, with tests and exact candidate paths; keeps refactors separate from score-affecting changes so gains remain attributable.
7.  **Independent Red-Team Reviewer:** Audits code for subtle bugs (e.g., augmentation semantics, metric calculation) before execution.

### Protocol Guardrails
*   **Agency & Authority Boundaries:** Agents can research, refactor, and propose, but do not self-certify their scores or launch external submissions.
*   **Strict Evidence Labels:** All claims in documentation must be explicitly tagged as `[VERIFIED]`, `[REPORTED]` (e.g., by another competitor, unverified locally), or `[HYPOTHESIS]`.
*   **No Concurrent Edits:** Agents must never edit the same files concurrently to prevent race conditions and lost context.
*   **Human Approval Gates:** No Kaggle submission or major compute job may be launched without explicit human review of the OOF justification and code diff.

## 6. Next-Competition Playbook (Day 1 to Endgame)

1.  **Rules & Data Inventory (Day 1):** Document metrics, constraints, data types, and potential leakage vectors.
2.  **Early Reproducible Baseline:** Build a simple, end-to-end training and inference pipeline immediately to establish a functioning metric baseline.
3.  **Correct Grouped Splits:** Implement rigorous cross-validation splitting based on patient ID, report hash, or equivalent grouping logic to prevent leakage. Verify splits mathematically.
4.  **Per-Study OOF Artifact & Provenance:** Establish a strict system for saving and tracking exact fold-excluded predictions for every model. Without this, ensembling is blind.
5.  **Target/Subgroup Error Audit:** Analyze OOF predictions to identify specific failure modes, weak classes, or challenging subgroups.
6.  **Ordered Experiment Backlog:** Maintain a prioritized list of hypotheses.
7.  **One-Factor Ablations:** Test hypotheses one at a time, holding all other variables constant. Require development OOF for screening and a frozen untouched confirmation split/fold before promotion; do not imply tuning and scoring on the same OOF rows is independent validation.
8.  **OOF-Only Stacking & Nested Confirmation:** Any meta-model or rank blender must be fitted *only* on valid development OOF predictions and evaluated on a frozen untouched confirmation split/fold.
9.  **Limited Public-LB Probes:** Use the public LB sparingly to verify the CV/LB correlation slope, not to select models.
10. **Exact-Candidate Packaging & Endgame:** Pre-plan the final ensemble slots. Ensure reproducibility and compliance of the exact inference notebooks.

## 7. Prioritized Checklist & References

### Core Checklist
- [ ] Are CV splits grouped correctly by patient/report?
- [ ] Is the evaluation metric exactly matching the competition implementation?
- [ ] Do we have a complete, exact OOF prediction cache for the current best model?
- [ ] Are all documentation claims explicitly labeled as verified, reported, or hypothesis?

### References
*   [RSNA Knee Abnormality Detection Competition Page](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection)
*   [Official Competition Rules](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/rules)
*   [Competition Leaderboard](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/leaderboard)
*   *Local Evidence:* `docs/experiments.md`, `docs/oof-provenance-audit.md`, `docs/rsna_prior_solution_transfer_review.md`
