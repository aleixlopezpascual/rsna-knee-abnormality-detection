# Final Submission Selection Decision

**As of:** 2026-10-22 12:39 UTC

**Status:** Portfolio recommendation finalized here; Kaggle final-submission selections have **not** been changed.

**Scope:** Choose at most two final submissions, balancing the main leaderboard against the Efficiency Prize.

## Final recommended portfolio

| Slot | Recommended selection | Why |
|---|---|---|
| **1 — Main performance anchor** | **Exp-18, ref `57316129`, public AUC `0.958`** | Keep the established anchor. Exp-22 (`57478743`) also displays 0.958, but its public-LB-only micro-calibration has no exact-stack CV and does not demonstrate a gain. |
| **2 — Main-track hedge** | **Exp-19 Golden Fusion, ref `57365961`, public AUC `0.957`** | Chosen as the explicit anchor-plus-5-fold-ASL Probit hedge. This is a conservative tie-break, not evidence that it beats Exp-20 privately. |

**Do not use Slot 2 for Exp-15 Efficiency Champion (`57174412`, public AUC `0.946`).** The preferred second slot remains the close main-track hedge, not the long-shot efficiency entry. This is a portfolio recommendation, not a Kaggle UI action.

Exp-19 and Exp-20 (`57410300`) both displayed 0.957 on the public leaderboard. No private scores or fold-excluded OOF for the exact stacked candidates distinguish them. I choose Exp-19 because it adds the separately trained five-fold ASL specialist with documented component-level rare-target validation. Exp-20's DINOv2 stream is more orthogonal on paper, but the provenance audit flags an unresolved mismatch: the checkpoint manifest identifies DINOv2-Small while the notebook constructs `vit_base_patch14_dinov2` and loads with `strict=False`; the run's “loaded” message does not prove compatible weights were applied. Neither whole blend has exact-stack OOF, so this is risk management, not measured superiority. Exp-22 is not promoted because its public tie and smaller weight adjustment provide no CV evidence of improvement or a strong independent hedge.

The live submission-list refresh at **2026-10-22 12:39 UTC** showed refs `57316129` and `57478743` at 0.958, refs `57365961` and `57410300` at 0.957, all `COMPLETE`; private scores were unavailable.

## Medal context and private-leaderboard shakeup

The full public leaderboard snapshot at **2026-10-22 12:39 UTC** contained 6,376 teams; Aleix López was **rank 1,007** at displayed AUC **0.958**. Kaggle's progression rules for competitions with 1,000+ teams set Bronze at the top 10% (rank 637 here), Silver at the top 5% (rank 318), and Gold at the top 10 plus 0.2% (22 teams here). The current public rank is therefore 370 places outside the Bronze cutoff. The rounded 0.958 score band spans ranks 252–1,851, so its three-decimal display hides a wide rank spread; it is not a medal-probability estimate.

For a private shakeup, retain Exp-18 as the highest-scoring anchor and use Exp-19 as the second, more target-complementary candidate. The public/private split can reorder candidates, but neither this portfolio nor a public tie guarantees a medal. The exact Exp-19 blend still lacks fold-excluded OOF, so it is a hedge rather than a proven private-LB improvement. Sources: [current public leaderboard](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/leaderboard), [Kaggle medal thresholds](https://www.kaggle.com/progression/competitions), and [OOF provenance audit](oof-provenance-audit.md).

## Why the Efficiency entry is a long shot

The latest available public Efficiency Leaderboard output from Ryan Holbrook’s Kaggle notebook was generated **2026-10-21 21:17 UTC** and downloaded on **2026-10-22 12:27 UTC**. Its 6,222-row snapshot listed:

- Aleix López / Exp-15: **rank 170 of 6,222**, public AUC **0.946** (submission date 2026-10-14).
- Snapshot top-three displayed public AUCs: **0.964, 0.961, and 0.961**.
- The Efficiency Prize awards only three prizes: **$7,000, $6,000, and $5,000**.

That is **167 rank positions** behind third place in that snapshot. The leaderboard ranks submissions using both predictive performance and runtime; its downloadable table shows rank and public AUC, not the complete efficiency score or runtime contribution. This is the latest available public snapshot, not the final private efficiency ranking, so the low-probability assessment is not a guarantee.

Exp-15 remains a strong fast artifact (documented at under 100 seconds and under 2 GB RAM), but its position in the latest available public snapshot does not justify choosing it over the main-track hedge for this final-slot decision.

## External-data eligibility caveat

This is **not a legal determination**. The Exp-18 kernel metadata attaches [`nartaa/rsna-knee-publication-swa-weights-20261007`](https://www.kaggle.com/datasets/nartaa/rsna-knee-publication-swa-weights-20261007), whose authors report training with 2,399 OAI knees. The Exp-15 Efficiency Champion metadata attaches the same weight dataset, so it is not an OAI-free hedge; the .957 alternatives also inherit the Exp-18 anchor.

The host's general guidance says resources requiring only registration and acceptance of licensing terms meet the accessibility criterion, while institutional agreements or institution-only access do not ([discussion #743416](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/743416)). NIMH describes OAI as Open Access, but the OAI-specific geo-access question remained unanswered in the latest forum review ([#749145](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/749145)); Kaggle staff declined to pre-approve specific datasets or model weights ([#747455](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/747455)). Public weights trained on competition data are eligible for awards, with their source datasets disclosed ([#744056](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/744056)); that answer does not settle OAI-trained checkpoints. Treat this as unresolved eligibility risk, not proof of disqualification. This caveat does not change the recommended pair; both candidates inherit the Exp-18 anchor.

## Official selection constraints

- Kaggle allows **up to two** final submissions for judging.
- An Efficiency Prize submission must be selected for the Leaderboard Prize or be automatically selected under Kaggle’s My Submissions rules, and it must beat the sample-submission benchmark on the private leaderboard.
- A submission may be eligible for both the Main and Efficiency prizes.
- Efficiency awards are based on the **private** efficiency ranking; the public efficiency leaderboard is updated during the competition and does not expose the full efficiency score.

Sources: [official competition rules](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/rules), [official efficiency evaluation](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/overview/efficiency-prize-evaluation), and [public efficiency leaderboard notebook](https://www.kaggle.com/code/ryanholbrook/rsna-knee-abnormalities-efficiency-lb/output).

## Final UI action (not applied)

The final recommendation is **Exp-18 (`57316129`) + Exp-19 (`57365961`)**. No further model experiment or speculative submission is recommended. The Kaggle final-selection UI remains unchanged; applying this pair there is a separate external action and requires explicit authorization.