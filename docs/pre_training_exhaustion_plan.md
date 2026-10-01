# Pre-Training Exhaustion Protocol & The "120% Certainty" Blueprint

**Repository:** RSNA Knee Abnormalities Detection (`rsna-knee-abnormality-detection`)  
**Lead ML Engineering Strategy**

---

## 🎯 Executive Philosophy: Why We Do Not Train Blind

In competitive medical imaging challenges, naive teams waste hundreds of GPU hours training standard architectures on standard data, only to match or underperform community baselines.

The RSNA Knee competition landscape already contains **~35 pre-trained model weights** across 4 distinct model families (DINOv2, RadImageNet ResNet50, 4-arm Raptor, and 4-way CoAtNet).

**The Grandmaster Mandate:** We do not launch heavy training runs until we have:
1. Exhausted every zero-training ensembling, routing, and post-processing gain.
2. Conducted a diagnostic error audit to identify the precise sub-pathologies that bottleneck the score.
3. Formulated a targeted hypothesis (loss function, sampling, or architecture) that guarantees an empirical edge over existing checkpoints.

---

## 📋 Phase 1: Zero-Training Optimization Roadmap

Before training a single weight from scratch, we can extract measurable leaderboard gains using our local workstation:

### 1. Target-Specific Routing Weight Optimization
- **Current Baseline:** Static community weights:
  - `Lateral Meniscus`: 100% CoAtNet
  - `Medial Meniscus`: 80% CoAtNet / 20% Transformer
  - `ACL`, `Lateral OA`, `Fracture`: 75% CoAtNet / 25% Transformer
  - Default: 60% CoAtNet / 40% Transformer
- **Actionable Optimization:** Formulate the routing weights as a continuous bounded optimization problem on local validation data:
  $$\max_{w_1, \dots, w_{12} \in [0, 1]} \frac{1}{12} \sum_{i=1}^{12} \text{ROC-AUC}_i\Big(w_i \cdot \text{Rank}(\hat{y}_{\text{CNN}, i}) + (1 - w_i) \cdot \text{Rank}(\hat{y}_{\text{Trans}, i})\Big)$$
- **Expected Gain:** $+0.002\text{--}0.005$ Macro-AUC without training.

### 2. Volumetric Window Density Tuning ($k_{\text{eval}}$)
- In the Raptor pipeline, changing window sampling density from $k=62$ to $k=94$ slices per study produced a $+0.003$ public LB gain with zero retraining.
- **Actionable Optimization:** Test $k_{\text{eval}} \in \{94, 112, 128\}$ with adaptive slice striding along the joint space, capturing finer meniscal boundaries.

### 3. Multi-Family Rank Space Ensembling
- Blend the predictions of our modular pipeline (`exp-01`) with the `0.946 SOTA` predictions (`sub-02`) using `src.ensemble.blend_routed_predictions`.
- Because our baseline uses different slice pooling and seed configurations, rank-averaging provides immediate error decorrelation.

### 4. Non-Linear Rank Calibration & Tail Spreading
- ROC-AUC on rare classes (`Fracture`, `MCL`) is extremely sensitive to tie-breaks in the upper decision decile.
- **Actionable Optimization:** Apply power rank stretching ($p_{\text{cal}} = p^{\gamma}$) or quantile smoothing on low-confidence predictions to minimize false-positive rank penalties.

---

## 🔬 Phase 2: Diagnostic Error Audit (Finding the Bottleneck)

A breakdown of the 12 targets across community models reveals a severe performance disparity:

| Category | Finding | Current Community AUC | Status | Action |
| :--- | :--- | :---: | :--- | :--- |
| **Saturated** | `Effusion` | ~0.970 | Near ceiling | **Do not train** (diminishing returns) |
| | `ACL` | ~0.965 | Near ceiling | **Do not train** |
| **Strong** | `PF OA` | ~0.950 | Well captured | Maintain existing weights |
| | `Baker's` | ~0.945 | Well captured | Maintain existing weights |
| | `Medial Meniscus` | ~0.940 | Strong | Maintain existing weights |
| | `Medial OA` | ~0.938 | Strong | Maintain existing weights |
| **The Bottlenecks** | `Lateral OA` | ~0.915 | Needs boost | Target for specialist |
| | `Lateral Meniscus` | ~0.905 | Severe error rate | **High ROI target** |
| | `Contusion` | ~0.890 | Low signal | **High ROI target** |
| | `MCL` | ~0.875 | Rare (<3% positive) | **High ROI target** |
| | `Fracture` | ~0.840 | Rare (<2% positive) | **Highest ROI target** |

**Conclusion:** The entire competition will be won or lost on the bottom 4 findings: `Fracture`, `MCL`, `Contusion`, and `Lateral Meniscus`. Generic models waste 70% of gradient updates reinforcing already-solved findings (`Effusion`, `ACL`).

---

## 🛡️ Phase 3: The 120% Certainty Training Blueprint

When we trigger model training, we will strictly follow this hypothesis-driven blueprint:

### The Hypothesis
> *Standard BCE loss on all 12 targets causes severe gradient starvation for `Fracture` and `MCL` due to massive class imbalance (>97% negative). By training a dedicated **Rare-Finding Specialist** using **Asymmetric Focal Loss** ($\gamma=2.0$, $\alpha_c \propto 1 / \sqrt{f_c}$) initialized from **RadImageNet ResNet50**, we can boost minority class AUC by $+0.03\text{--}0.06$ without degrading existing ensemble predictions.*

### The Validation Gate (Go / No-Go Decision Criteria)
Before committing to full 5-fold training:
1. **Fold 0 Smoke Check:** Train Fold 0 on Kaggle for 5 epochs (~45 minutes).
2. **Success Criterion:** The specialist model's out-of-fold AUC on `Fracture` and `MCL` must exceed the current baseline by at least **$+0.02$ AUC**.
3. **Blend Verification:** Blending the specialist predictions into our 0.946 ensemble must increase the overall Macro-AUC on Fold 0 validation.
4. **Execution Decision:** Only if the Gate passes do we trigger Folds 1–4.
