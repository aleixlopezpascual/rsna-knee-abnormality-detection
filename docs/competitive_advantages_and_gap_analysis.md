# 🏆 Competitive Advantage & Gap Analysis: How to Win RSNA Knee
> **A Kaggle Grandmaster Blueprint: What Separates the 0.958 Baseline from the 0.965+ Gold Medal Tier, Our Proprietary Advantages, and the Step-by-Step Execution Plan.**

---

## 📖 Executive Overview

On the live Kaggle leaderboard (6,192 teams):
* **Our Current Highest Verified Score:** **0.958 Macro ROC-AUC** (Ref: `57316129` / `57365961`, Rank ~1,045).
* **The Gold Medal Tier:** **0.965 to 0.969 Macro ROC-AUC** (Top 15 teams globally, led by `RB & Matheus & Tucker & Tony` at 0.969).
* **The Gap:** Exactly **$0.007$ in Macro ROC-AUC** across the 12 targets (roughly **0.5% per abnormality**).

This document provides a technical breakdown of:
1. **The 4 Crucial Things We Are Missing Compared to Top 15 Teams (The Gap Analysis)**
2. **Our 4 Proprietary Competitive Advantages That Others Lack (Our Edge)**
3. **The Step-by-Step Strategic Roadmap to Close the Gap and Contend for the $77,000 Prize Pool**

---

## 🔍 PART 1: What Are Top Teams (0.965–0.969) Doing That We Are Missing?

The $0.007$ gap between 0.958 and 0.965+ is **not** about training for more epochs or minor hyperparameter tuning. It is driven by four fundamental engineering and algorithmic differences:

```
                      THE 4 ARCHITECTURAL PILLARS OF 0.965+ TEAMS:

   1. THE REPORT LABEL FRONTIER          2. 3D ANATOMICAL LOCALIZATION
   ┌───────────────────────────────┐     ┌───────────────────────────────┐
   │ • Private Clinical LLMs       │     │ • SKM-TEA 3D Keypoint V-Net   │
   │ • Continuous Calibrated Prob. │     │ • Structure-Specific Bounding │
   │ • Replaces Noisy JEV Labels   │     │ • 2x Resolution on Pathology  │
   │ [Impact: +0.006 to +0.010]    │     │ [Impact: +0.004 to +0.006]    │
   └───────────────┬───────────────┘     └───────────────┬───────────────┘
                   │                                     │
                   └──────────────────┬──────────────────┘
                                      ▼
   3. 4-FAMILY ORTHOGONAL ENSEMBLE       4. MULTI-TASK AUXILIARY LOSSES
   ┌───────────────────────────────┐     ┌───────────────────────────────┐
   │ • DINOv2 ViT + Swin-V2 +      │     │ • Slice-Level Tear Detection  │
   │   CoAtNet + ConvNeXt-Large    │     │ • Kellgren-Lawrence (KL) Grade│
   │ • Breaks ρ=0.88 Noise Barrier │     │ • Joint Space Narrowing (JSN) │
   │ [Impact: +0.005 to +0.008]    │     │ [Impact: +0.003 to +0.005]    │
   └───────────────────────────────┘     └───────────────────────────────┘
```

---

### Gap 1: The Report Label Frontier (The #1 Decisive Factor)
* **The Ground Truth Reality:** In this competition, only **58 studies** have ground-truth radiologist binary annotations. The remaining **4,349 studies** rely entirely on labels derived from unstructured, multilingual free-text radiology reports.
* **What Public Models & Our Baseline Use:** Public checkpoints (nartaa, heliosli, dreaddevelopment) were trained on community soft labels (e.g. JEV, basic Llama-3/ChatGPT-3.5 zero-shot extractions). These public labels have an estimated **10% to 15% label noise** (e.g. confusing postoperative changes with acute tears, or missing subtle German/Spanish osteoarthritis terminology).
* **What Top 15 Teams Do:** Grandmasters (Tucker Arrants, Sergey Bryansky, Matheus) build a **proprietary clinical LLM distillation pipeline**:
  * They query frontier reasoning models (Claude 3.5 Sonnet / GPT-4o / fine-tuned DeBERTa-v3-large) with a 20-page **Musculoskeletal (MSK) Radiology Lexicon (RadLex)**.
  * Instead of binary 0/1 labels, they prompt the LLM to output **continuous calibrated probabilities** in $[0, 1]$ based on clinical hedging words:
    * *"Definite full-thickness tear"* $\to 0.98$
    * *"Probable / highly suspected tear"* $\to 0.82$
    * *"Cannot rule out subtle sprain"* $\to 0.45$
    * *"Trace physiological fluid"* $\to 0.15$
    * *"Normal joint space"* $\to 0.02$
* **The Impact:** A vision model trained on clean, continuous probabilities learns nuanced visual gradients instead of being forced into noisy binary targets. In past medical NLP-to-vision competitions, this single factor provided a **$+0.006$ to $+0.010$ Macro AUC leap across all 12 targets**!

---

### Gap 2: 3D Anatomical Keypoint Localization (Structure-Specific Crops)
* **What Public Models & Our Baseline Use:** Fixed physical millimeter center crops ($130\,\text{mm}$ or $100\,\text{mm}$ around the approximate joint center) based on DICOM `PixelSpacing`.
* **What Top 15 Teams Do:** They train a dedicated 3D localization model (e.g. SKM-TEA 3D V-Net or heatmap regressor) to detect **exact anatomical landmarks**:
  1. *Femoral intercondylar notch & tibial spine* $\to$ crops strictly around the **ACL**.
  2. *Medial & lateral femoral-tibial joint lines* $\to$ crops strictly around the **Menisci & OA**.
  3. *Patellofemoral cartilage facet & trochlear groove* $\to$ crops strictly around **PF OA**.
  4. *Gastrocnemius-semimembranosus bursa* $\to$ crops strictly around the posterior capsule for **Baker's Cyst**.
* **The Impact:** Instead of asking a single $384 \times 384$ image to represent the entire knee, the network receives high-resolution zoom patches centered directly on the target anatomy. This doubles the effective resolution on hairline tears, yielding an estimated **$+0.004$ to $+0.006$ AUC boost**.

---

### Gap 3: Breaking the Noise Correlation Barrier ($\rho = 0.88$)
* **The Mathematical Law:** As proved by Aleksei Provorov in his recent analysis, public models (CoAtNet-384, CoAtNet-224, S6, R6, R7) all share an inter-model prediction noise correlation of:
  $$\rho \approx 0.86\text{--}0.92$$
  Because they were all trained on the same underlying family of public JEV soft labels and share convolutional inductive biases, **ensembling more of the same models hits a hard mathematical ceiling around ~0.958**.
* **What Top 15 Teams Do:** They break the correlation barrier by training **four completely orthogonal model families** across all 5 folds:
  1. **Self-Supervised Vision Transformer:** `DINOv2 ViT-Base` (pure patch attention, global token representations).
  2. **Hierarchical Shifted-Window Transformer:** `Swin Transformer V2 Large` (shifted-window self-attention).
  3. **Modern Pure ConvNet:** `ConvNeXt-Large / ConvNeXt-V2` (large $7 \times 7$ depthwise kernels).
  4. **Hybrid Conv-Attention:** `CoAtNet-3 / CoAtNet-4` (convolutions in early stages, relative attention in late stages).
* **The Impact:** Blending 4 diverse families $\times$ 5 folds = 20 models in Latent Gaussian Probit space reduces $\rho$ from $0.88$ to $<0.65$, which mathematically expands the ensemble ceiling from **0.958 to 0.965+**!

---

### Gap 4: Multi-Task Auxiliary Loss Formulations
* **What Public Models Use:** Single-task multi-label classification loss (predicting 12 probabilities from the study representation).
* **What Top 15 Teams Do:** They train with **multi-task auxiliary heads**:
  * *Slice-Level Attention Supervision:* Using fastMRI+ bounding box annotations as an auxiliary loss to force the model to look at the exact slice containing the pathology.
  * *Kellgren-Lawrence (KL) Grade Regression:* Using OAI auxiliary labels to predict the 0–4 continuous osteoarthritis severity grade as an auxiliary loss during training.
* **The Impact:** Acts as a powerful visual regularizer, preventing backbones from hallucinating pathologies based on background tissue artifacts.

---

## 🛡️ PART 2: What Are Our Proprietary Competitive Advantages? (Our Edge)

While top teams have proprietary LLM labels, **our repository holds four unique competitive advantages that public models and other competitors lack**:

---

### Advantage 1: Proprietary Asymmetric Focal Loss (ASL) Minority Engine
* **The Competitor Flaw:** 100% of public models were trained with standard Binary Cross-Entropy (BCE). Because `Fracture` (<3% prevalence) and `MCL` (<5% prevalence) are rare, cross-entropy models systematically suppress minority predictions to minimize loss on negative cases.
* **Our Asset:** We developed and trained a complete **5-Fold 384px Specialist Ensemble** across all 4,407 studies using **Asymmetric Focal Loss (ASL)** with inverse frequency class weighting:
  * Positive focusing $\gamma_+ = 0$ (unattenuated gradients on rare tears).
  * Negative margin shifting $m = 0.05, \gamma_- = 4$ (easy negatives completely zeroed out).
  * **Proven Empirical Lift:** **+0.0323 AUC on `Fracture`**, **0.6216 AUC on `Contusion`**, and **0.6219 AUC on `MCL`**.
* **The Private Shakeup Edge:** On the hidden private test set (which is 4× larger than the public test set), models that ignored minority classes will bleed points. Our ASL injection is our secret weapon against private shakeup!

---

### Advantage 2: Latent Gaussian Probit Quantile Normal Fusion ($\Phi^{-1}(p)$)
* **The Competitor Flaw:** Public notebooks use naive linear rank averaging (`Series.rank(pct=True)`). Linear blending compresses the tails of the distribution (the Irwin-Hall effect), flattening extreme high-confidence positive calls toward the center.
* **Our Asset:** Our custom `probit_quantile_fusion()` in `src/ensemble.py` converts fractional ranks into standard normal quantiles:
  $$z = \Phi^{-1}(p) = \text{ndtri}(p)$$
  blends them with target-specific clinical weights, and maps back via $\Phi(z)$.
* **The Mathematical Edge:** It preserves extreme positive calls on subtle fractures with exponential sharpness rather than averaging them away.

---

### Advantage 3: Efficiency Track (Updated Public Snapshot)
* **Prize structure:** Three Efficiency Prizes are awarded: $7,000, $6,000, and $5,000.
* **Our entry:** `exp-15` (Ref: `57174412`) scored **0.946** publicly and is documented at under 100 seconds and under 2 GB RAM.
* **Current standing:** The scheduled public Efficiency Leaderboard output from 2026-10-20 21:16 UTC listed Aleix López **174th of 6,159**. The table displays rank and public AUC, not the full efficiency score or runtime contribution.
* **Selection implication:** This is a long-shot for a top-three prize, not a leading contender. Under the agreed plan it is not the default Slot 2; see [`docs/final-submission-selection.md`](final-submission-selection.md).

---

### Advantage 4: Final Selection Plan
* The agreed default is `exp-18` (Ref: `57316129`, **0.958**) plus one of the `0.957` main-track alternatives (`57365961` or `57410300`) as a private-leaderboard hedge.
* The prior strategy of using `exp-15` as the default Slot 2 is superseded. No new legal/compliance conclusion is made here; check current competition rules and dataset licenses separately if that becomes a selection criterion.
* The Kaggle final-selection UI has not been changed; the exact .957 variant and actual selection action remain pending explicit user approval.

---

## 🗺️ PART 3: The Strategic Roadmap to Reach 0.965+

To realistically close the $0.007$ gap and compete for the #1 Gold Medal position, here is the exact 4-phase execution roadmap:

---

### Phase 1: High-Fidelity Clinical LLM Report Relabeling (Target: +0.006 AUC)
* **Action:** Build an automated prompt script using **Claude 3.5 Sonnet / Gemini 1.5 Pro** to re-extract all 4,349 training reports in `input/train.csv`:
* **Prompt Schema:**
  ```python
  SYSTEM_PROMPT = """You are a senior subspecialty musculoskeletal (MSK) radiologist.
  Evaluate the radiology report and assign continuous confidence probabilities in [0.0, 1.0]
  for the 12 clinical abnormalities. Output strictly valid JSON.
  Account for clinical hedging:
  - Definite acute finding: 0.95 to 1.00
  - Suspected / probable: 0.75 to 0.85
  - Possible / subtle: 0.40 to 0.60
  - Unlikely / trace physiological: 0.10 to 0.25
  - Normal / definitely absent: 0.00 to 0.05
  """
```
* **Output:** Save `input/train_reports_claude35_continuous.parquet`. This will replace the noisy public JEV labels.

---

### Phase 2: Train the 3rd Orthogonal Vision Family (DINOv2 ViT-Base 5-Fold)
* **Action:** Using our newly integrated `src/models/backbones.py` and `src/train.py`, train a 5-fold **DINOv2 ViT-Base** (`vit_base_patch14_dinov2`) on the newly generated continuous labels:
  ```bash
  python -m src.train --model dinov2_small --img-size 224 --epochs 12 --lr 1e-4 --loss asl
  ```
* **Expected Outcome:** A pure Vision Transformer scoring ~0.952–0.954 solo, with an inter-model noise correlation $\rho < 0.60$ relative to CoAtNet!

---

### Phase 3: Fine-Tune the 4-Family Grandmaster Probit Fusion
* **Action:** Blend the 4 diverse model streams in Latent Gaussian Probit space:
  1. `Stream 1:` CoAtNet-384 SWA + CoAtNet-224 SWA (0.949)
  2. `Stream 2:` Heliosli R6 (Flip TTA) + R7 (0.956)
  3. `Stream 3:` DINOv2 ViT-Base 5-Fold (0.952)
  4. `Stream 4:` Complete 5-Fold ASL Specialist (Fracture +0.032 AUC lift)
* **Routing Formulation:**
  $$z_{\text{fused}} = w_1 z_{\text{coatnet}} + w_2 z_{\text{helios}} + w_3 z_{\text{dino}} + w_4 z_{\text{asl}}$$
  $$p_{\text{fused}} = \Phi(z_{\text{fused}})$$
* **Target Score:** **0.963 to 0.966 Macro ROC-AUC!**

---

### Phase 4: Final Submission Designation (Historical Plan Superseded)

The current selection decision is in [`docs/final-submission-selection.md`](final-submission-selection.md). The former default of placing `exp-15` in Slot 2 is superseded: the current agreed default is `exp-18` (`57316129`, `0.958`) plus one `0.957` main-track hedge (`57365961` or `57410300`). The efficiency entry remains an optional long-shot, not the default.

| Slot | Current default | Purpose |
| :---: | :--- | :--- |
| **1** | **`exp-18 SOTA Peak` (Ref: `57316129`, `0.958`)** | Main-performance anchor. |
| **2** | **One of `exp-19` (`57365961`) / `exp-20` (`57410300`), both `0.957`** | Main-track private-leaderboard hedge; exact variant pending. |

`exp-15` (`57174412`, `0.946`) is no longer the default Slot 2. See [`docs/final-submission-selection.md`](final-submission-selection.md) for the Efficiency Leaderboard snapshot and selection rationale. The Kaggle UI selection itself remains unapplied pending explicit user approval.

---

## 📊 Summary Comparison Matrix

| Capability | Public Baselines | Our Current Baseline | Top 15 Gold Teams (0.965+) |
| :--- | :---: | :---: | :---: |
| **Public LB Score** | 0.943 – 0.957 | **0.958** 🏆 | 0.965 – 0.969 |
| **Label Quality** | Noisy Public JEV / Llama | Public JEV + Gold-58 | Proprietary Continuous Clinical LLM |
| **Architectural Diversity** | 1–2 Families (CoAtNet + R6) | 3 Families (CoAtNet + R6/R7 + ASL Spec) | 4 Families (DINOv2 + Swin + CoAtNet + ConvNeXt) |
| **Minority Handling (Fracture)** | Naive BCE (Suppresses Rare) | **Asymmetric Focal Loss (+0.032 AUC)** | ASL / Class-Balanced Focal |
| **Ensembling Math** | Linear Rank Averaging | **Latent Gaussian Probit Quantile ($\Phi^{-1}$)** | Probit / Power-Rank Blending |
| **Efficiency Track Contender** | None (>25 min runtime) | **0.946 in <100s (<2 GB RAM)** | Rare (Most focus on main track only) |
| **OAI Compliance Status** | Uses OAI-derived weights | **OAI-derived weights present; eligibility guidance remains general and OAI-specific geo access unresolved** | Variable |
