# Final Submission Selection Decision

**As of:** 2026-10-21 00:21 CEST

**Status:** Strategy agreed in chat; Kaggle final-submission selections have **not** been changed.

**Scope:** Choose at most two final submissions, balancing the main leaderboard against the Efficiency Prize.

## Agreed default portfolio

| Slot | Default choice | Why |
|---|---|---|
| **1 — Main performance anchor** | **Exp-18, ref `57316129`, public AUC `0.958`** | Highest verified public score in the current submission history. Keep this as the main entry. |
| **2 — Main-track hedge** | **One of Exp-19 `57365961` or Exp-20 `57410300`, both public AUC `0.957`** | Preserve a close alternate for private-leaderboard variation. The exact .957 entry has not yet been chosen. |

**Default: do not use Slot 2 for Exp-15 Efficiency Champion (`57174412`, public AUC `0.946`).** User agreed that the .957 main-track alternative is the preferred second selection over the current efficiency entry. This is a portfolio recommendation, not a Kaggle UI action.

## Why the Efficiency entry is a long shot

The scheduled public Efficiency Leaderboard output downloaded from Ryan Holbrook’s Kaggle notebook on **2026-10-20 21:16 UTC** listed:

- Aleix López / Exp-15: **rank 174 of 6,159**, public AUC **0.946** (submission date 2026-10-14).
- Top three displayed public AUCs: **0.961, 0.961, and 0.965**.
- The Efficiency Prize awards only three prizes: **$7,000, $6,000, and $5,000**.

That is **171 rank positions** from the current third-place position. The leaderboard ranks submissions using both predictive performance and runtime; its downloadable table shows rank and public AUC, not the complete efficiency score or runtime contribution. The public board is a snapshot, not the final private efficiency ranking, so this is a low-probability assessment—not a guarantee.

Exp-15 remains a strong fast artifact (documented at under 100 seconds and under 2 GB RAM), but its current public efficiency position does not justify further tuning time under the agreed plan.

## External-data eligibility caveat

This is **not a legal determination**. The Exp-18 kernel metadata attaches [`nartaa/rsna-knee-publication-swa-weights-20261007`](https://www.kaggle.com/datasets/nartaa/rsna-knee-publication-swa-weights-20261007), whose authors report training with 2,399 OAI knees. The Exp-15 Efficiency Champion metadata attaches the same weight dataset, so it is not an OAI-free hedge; the .957 alternatives also inherit the Exp-18 anchor.

The host's general guidance says resources requiring only registration and acceptance of licensing terms meet the accessibility criterion, while institutional agreements or institution-only access do not ([discussion #743416](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/743416)). NIMH describes OAI as Open Access, but the OAI-specific geo-access question remained unanswered in the latest forum review ([#749145](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/749145)); Kaggle staff declined to pre-approve specific datasets or model weights ([#747455](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/747455)). Public weights trained on competition data are eligible for awards, with their source datasets disclosed ([#744056](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/744056)); that answer does not settle OAI-trained checkpoints. Treat this as unresolved eligibility risk, not proof of disqualification. The selection plan above has not changed.

## Official selection constraints

- Kaggle allows **up to two** final submissions for judging.
- An Efficiency Prize submission must be selected for the Leaderboard Prize or be automatically selected under Kaggle’s My Submissions rules, and it must beat the sample-submission benchmark on the private leaderboard.
- A submission may be eligible for both the Main and Efficiency prizes.
- Efficiency awards are based on the **private** efficiency ranking; the public efficiency leaderboard is updated during the competition and does not expose the full efficiency score.

Sources: [official competition rules](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/rules), [official efficiency evaluation](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/overview/efficiency-prize-evaluation), and [public efficiency leaderboard notebook](https://www.kaggle.com/code/ryanholbrook/rsna-knee-abnormalities-efficiency-lb/output).

## Final action still pending

The agreed default is **Exp-18 + one .957 main-track hedge**. Before changing Kaggle’s final-selection UI, choose between `57365961` and `57410300` and explicitly authorize the selection. If the priority later changes to pursuing the Efficiency Prize as a long shot, Slot 2 can instead be `57174412`; that trade-off has not been made.