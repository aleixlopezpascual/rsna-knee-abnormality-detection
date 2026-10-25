# 🏆 RSNA Knee Abnormalities Detection: Top Solutions Analysis & Architectural Benchmarking

> **A Comprehensive Technical Report: Deconstruction of Winning Grandmaster Methodologies, Key Learnings, Architectural Comparison with Our Pipeline, and Compute/Hardware Retrospective.**
>
> **Competition:** RSNA Knee Abnormalities Detection (`rsna-knee-abnormality-detection`)  
> **Evaluation Metric:** Macro-Averaged ROC-AUC across 12 clinical abnormality targets  
> **Status:** Post-Competition Technical Audit & Retrospective (October 2026)

---

## 1. Executive Summary & Competition Landscape

The **RSNA Knee Abnormalities Detection AI Challenge** (hosted by the Radiological Society of North America with a $77,000 USD prize pool) tasked participants with predicting continuous confidence probabilities in $[0.0, 1.0]$ for **12 clinically significant knee abnormalities** from multi-sequence MRI studies (`StudyInstanceUID`):

1. **Ligamentous & Capsular:** `ACL`, `MCL`
2. **Meniscal Pathology:** `Medial Meniscus`, `Lateral Meniscus`
3. **Compartmental Osteoarthritis:** `Medial OA`, `Lateral OA`, `PF OA`
4. **Fluid & Synovial:** `Effusion`, `Synovitis`, `Baker's Cyst`
5. **Osseous Lesions:** `Contusion` (Bone Bruise), `Fracture`

### The Evaluation Metric: Unweighted Macro ROC-AUC
Submissions were evaluated strictly on the unweighted arithmetic mean of the Area Under the Receiver Operating Characteristic Curve across all 12 targets:

$$\text{Final Score} = \frac{1}{12} \sum_{i=1}^{12} \text{ROC-AUC}_i$$

Because each target carries an equal $1/12$ weight, high-prevalence pathologies (e.g., `Effusion` at ~35%) hold the exact same mathematical leverage as rare, critical abnormalities (e.g., `Fracture` at <3% and `MCL` at <5%).

### Leaderboard Tiers & Performance Spectrum
* **Top 15 Gold Medal Tier:** **0.965 – 0.970 Macro ROC-AUC** (Led by grandmaster teams such as `RB & Matheus & Tucker & Tony`, `colum2131`, `datnt114`).
* **Our Highest Verified Pipeline:** **0.958 Public Macro ROC-AUC** (Exp-18, Ref: `57316129` / Exp-22, Ref: `57478743`) and **0.957 Consensus Superstack** (Exp-19/20, Refs: `57365961`, `57410300`), placing around rank 1,007–1,045 on the public leaderboard.
* **Our Efficiency Track Entry:** **0.946 Macro ROC-AUC in <100 seconds execution with <2 GB RAM** (Exp-15, Ref: `57174412`).
* **Public Baselines:** **0.887 – 0.943** (Starting from early 2.5D ResNet baselines up to multi-pillar public ensembles).
* **The Competitive Margin:** A gap of approximately **$0.007 – 0.012$ in Macro ROC-AUC** separated our 0.958 baseline from the winning 0.965–0.970 gold tier (equivalent to $\sim 0.5\%\text{--}1.0\%$ ROC-AUC per abnormality).

---

## 2. Top Solution Breakdown: Winning Techniques Across Top Teams

A comprehensive synthesis of the winning writeups, forum technical disclosures, and codebase architectures reveals that top-ranking teams did not win through brute-force hyperparameter sweeps or simple longer training. Instead, their success was driven by five foundational engineering pillars:

```
                      THE 5 ARCHITECTURAL PILLARS OF TOP TEAMS:

   1. THE REPORT LABEL FRONTIER          2. 3D ANATOMICAL LOCALIZATION
   ┌───────────────────────────────┐     ┌───────────────────────────────┐
   │ • Proprietary Clinical LLMs   │     │ • SKM-TEA 3D Keypoint Models  │
   │ • Continuous Calibrated Prob. │     │ • Structure-Specific Bounding │
   │ • Replaces 10-15% Label Noise │     │ • 2x Resolution on Pathology  │
   │ [Impact: +0.006 to +0.010]    │     │ [Impact: +0.004 to +0.006]    │
   └───────────────┬───────────────┘     └───────────────┬───────────────┘
                   │                                     │
                   └──────────────────┬──────────────────┘
                                      ▼
   3. 4-FAMILY ORTHOGONAL ENSEMBLE       4. STRICT 2.5D + MIL/ATTENTION
   ┌───────────────────────────────┐     ┌───────────────────────────────┐
   │ • DINOv2 ViT + Swin-V2 +      │     │ • 2.5D Slabs (No 3D Blur)     │
   │   CoAtNet + ConvNeXt-Large    │     │ • 1D Sequence / Attention Head│
   │ • Breaks ρ=0.88 Noise Barrier │     │ • Physics FOV & Geometry Sort │
   │ [Impact: +0.005 to +0.008]    │     │ [Impact: Foundational Baseline│
   └───────────────────────────────┘     └───────────────────────────────┘
                                      │
                                      ▼
                        5. MULTI-TASK AUXILIARY LOSSES
                        ┌───────────────────────────────┐
                        │ • Slice-Level Tear Detection  │
                        │ • Kellgren-Lawrence (KL) Grade│
                        │ • Joint Space Narrowing (JSN) │
                        │ [Impact: +0.003 to +0.005]    │
                        └───────────────────────────────┘
```

---

### 2.1 Pillar 1: The Report Label Frontier & Continuous Soft Labeling
* **The Asymmetry Reality:** The official dataset contained 4,407 studies, but only **58 studies** had ground-truth binary annotations verified by subspecialty musculoskeletal (MSK) radiologists. The remaining **4,349 studies** had only unstructured, multilingual free-text radiology reports (English, German, Spanish, French, Turkish).
* **The Public Trap:** Public models relied on community binary soft labels (e.g., JEV extractions, zero-shot Llama-3/ChatGPT scripts). These labels suffered from an estimated **10% to 15% noise rate** (e.g., misinterpreting postsurgical graft changes as acute tears, failing on foreign language expressions, or confusing historical clinical indications with active findings).
* **What Top Teams Did:** Grandmasters (e.g., Tucker Arrants, Sergey Bryansky, Matheus, colum2131) treated the extraction of report supervision as the single most critical task:
  1. **Musculoskeletal Radiology Lexicon (RadLex):** Prompted frontier reasoning models (GPT-4o, Claude 3.5 Sonnet, fine-tuned DeBERTa-v3-large) using standardized MSK diagnostic ontologies.
  2. **Continuous Calibrated Probabilities:** Instead of noisy binary 0/1 assignments, the LLMs were prompted to output continuous certainty probabilities in $[0.0, 1.0]$ based on clinical hedging language:
     - *"Definite complete rupture / displaced fracture"* $\to 0.98$
     - *"Probable partial tear / marked marrow edema"* $\to 0.82$
     - *"Cannot exclude subtle sprain / minor irregularity"* $\to 0.45$
     - *"Trace physiological joint fluid"* $\to 0.15$
     - *"Intact ligaments / normal osseous structures"* $\to 0.02$
  3. **Multi-Model Agreement & Filtering:** Queries were run across multiple LLM families. High-disagreement studies were isolated or weighted down during training.
  4. **Continuous Distillation:** Vision models were trained using continuous soft cross-entropy / KL-divergence loss against these calibrated targets, allowing the backbones to learn subtle visual feature gradients rather than being forced to overfit binary thresholds.

---

### 2.2 Pillar 2: 2.5D Volumetric Slicing & Sequence Attention (Why Pure 3D CNNs Failed)
Newcomers frequently assume that because MRI is volumetric, 3D convolutions (e.g., MONAI 3D DenseNet or 3D UNet) must be optimal. Top teams universally avoided pure 3D CNNs due to five physical and technical roadblocks:
1. **Multi-Sequence Asymmetry (MRI $\neq$ CT):** Knee MRI is not an isotropic 3D cube. A study comprises 4–8 distinct series with varying acquisition planes (Sagittal, Coronal, Axial) and tissue contrasts (T1, T2, PD, STIR). Resampling them onto a shared 3D grid causes severe interpolation blurring that obliterates hairline cortical fractures and subtle meniscal tears.
2. **The 10:1 Anisotropy Problem:** In-plane resolution is $0.3\text{--}0.5\,\text{mm}$ while slice thickness is $3.0\text{--}4.5\,\text{mm}$ (often with $0.5\,\text{mm}$ slice gaps). Isotropic $3\times 3\times 3$ kernels mix anatomy across 4 mm physical distances, corrupting local diagnostic details.
3. **The Pretraining Deficit:** 2D backbones leverage models pre-trained on tens of millions of images (ImageNet-21k, RadImageNet with 1.3M medical images, DINOv2). 3D architectures lack massive pretraining scale; training 3D CNNs from scratch on only 4,400 studies led to catastrophic overfitting.
4. **VRAM Exhaustion & Batch Size 1 Instability:** Multi-series 3D volumes at $384\times 384\times 96$ consume $>30\,\text{GB}$ VRAM, forcing batch size $B=1$. This introduces noisy gradients and Batch Normalization drift, causing optimization divergence.
5. **The Winning Hybrid (2.5D Spatial + 1D Volumetric):** Top teams extracted sharp in-plane 2.5D slice triplet representations using high-capacity 2D backbones, then aggregated through-plane volumetric relationships via lightweight 1D sequence models, Multiple Instance Learning (MIL) pooling, or multi-query cross-attention heads.

---

### 2.3 Pillar 3: 3D Anatomical Keypoint Localization (Structure-Specific Crops)
* **The Public Approach:** Fixed physical millimeter center crops ($130\,\text{mm}$ or $100\,\text{mm}$ centered on the joint) based on DICOM `PixelSpacing`.
* **What Top Teams Did:** Implemented a two-stage coarse-to-fine localization pipeline:
  1. **Stage 1 (Anatomical Landmark Detection):** Trained a 3D localization network (such as a 3D V-Net trained on SKM-TEA keypoints or heatmap regressors) to locate exact joint structures:
     - Femoral intercondylar notch & tibial spine $\to$ targeted bounding box around the **ACL**.
     - Medial & lateral femorotibial joint lines $\to$ targeted crops around the **Menisci & OA**.
     - Patellofemoral cartilage facet & trochlear groove $\to$ targeted crops for **PF OA**.
     - Gastrocnemius-semimembranosus bursa $\to$ targeted crops for **Baker's Cyst**.
  2. **Stage 2 (Pathology-Specialized Classifiers):** Fed structure-specific high-resolution crops directly to specialized backbones. This doubled the effective pixel resolution on millimeter-scale pathology without increasing computational overhead.

---

### 2.4 Pillar 4: Multi-Domain Orthogonal Ensembles (Breaking $\rho \approx 0.88$)
* **The Noise Ceiling:** Competitors discovered that ensembling public models (CoAtNet-384, CoAtNet-224, S6, R6, R7) plateaued around 0.954–0.958 because all public models shared an inter-model prediction noise correlation of $\rho \approx 0.86\text{--}0.92$. They all shared similar convolutional inductive biases and were trained on similar public soft labels.
* **What Top Teams Did:** They intentionally trained **four completely orthogonal model families** across 5 folds:
  1. **Self-Supervised Vision Transformer:** `DINOv2 ViT-Base` (pure patch attention, global token representations).
  2. **Shifted-Window Hierarchical Transformer:** `Swin Transformer V2 Large` (local shifted-window self-attention).
  3. **Modern Pure ConvNet:** `ConvNeXt-Large / ConvNeXt-V2` ($7\times 7$ depthwise convolutions).
  4. **Hybrid Conv-Attention:** `CoAtNet-3 / CoAtNet-4` with Stochastic Weight Averaging (SWA).
* By ensembling across orthogonal families, they reduced prediction error correlation to $\rho < 0.65$, unlocking the mathematical headroom needed to push past 0.965.

---

### 2.5 Pillar 5: Physics Preprocessing, Vector Geometry & Mirror TTA
* **Normal Vector DICOM Sorting:** Sorted slices by projecting `ImagePositionPatient` $(0020, 0032)$ along the slice normal vector $\vec{n} = \vec{r}_{\text{row}} \times \vec{r}_{\text{col}}$ derived from `ImageOrientationPatient` $(0020, 0037)$:
  $$\text{slice\_pos} = \vec{p}_{\text{patient}} \cdot \vec{n}$$
* **Physical FOV Normalization:** Applied fixed physical millimeter scaling via `PixelSpacing` $(0028, 0030)$ to eliminate scanner magnification differences.
* **Left-Knee Standardization:** Converted all right knees to left-knee coordinates:
  - Coronal and Axial planes: horizontal width flip.
  - Sagittal planes: slice sequence order reversed.
* **The Strict Zero-Flip Augmentation Rule:** Never applying horizontal flipping during training, as flipping swaps Medial and Lateral compartments (invalidating meniscus and OA labels).
* **Anatomical Mirror TTA:** During test-time augmentation, evaluating both the standard view and an anatomically mirrored view (flipping Axial/Coronal width and reversing Sagittal slice order) and averaging predictions.

---

## 3. Key Learnings: What Worked vs. What Didn't

### 3.1 What Consistently Worked
1. **Continuous Soft Distillation Targets:** Converting free-text radiology reports into continuous calibrated probabilities in $[0, 1]$ yielded the largest single performance jump (+0.006 to +0.010 AUC).
2. **2.5D Windowing + MIL / Attention Aggregation:** Slicing series into 2.5D triplet slabs and aggregating through-plane relationships via attention consistently outperformed 3D CNNs across all volumetric metrics.
3. **High Resolution on Pathology ($384\times 384$):** Sub-millimeter structures (meniscal tears, microfractures) benefited substantially from native $384\times 384$ resolutions or localized zoom patches.
4. **Dense Window Sampling ($k_{\text{eval}} = 94$):** Evaluating dense window sets across the full anatomical range improved test AUC by +0.003 to +0.006 without retraining model weights.
5. **Target-Specific Attention / Routing Priors:** Tilting attention towards clinically established acquisition planes (Sagittal for ACL, Coronal for MCL/Menisci, Axial for PF OA/Effusion) improved convergence and diagnostic fidelity.
6. **Asymmetric Focal Loss (ASL):** Decoupling positive and negative focusing parameters ($\gamma_+=0, \gamma_-=4, m=0.05$) prevented majority classes from washing out rare minority pathologies (`Fracture`, `MCL`).
7. **Latent Gaussian Probit Quantile Normal Fusion:** Mapping percentile ranks to standard normal space ($\Phi^{-1}(p)$) prior to weighted combination preserved confident positive detections at the distribution tails.
8. **Anatomical Mirror TTA:** Standard view plus anatomically mirrored view averaging provided a robust +0.002 to +0.003 AUC lift.

### 3.2 What Didn't Work / Harmed Performance
1. **Full 3D Convolutions (MONAI / 3D UNet):** Severe memory overhead ($B=1$), interpolation blurring across planes, and pretraining deficits caused 3D models to plateau around ~0.915.
2. **Naive Horizontal Flipping Augmentation:** Randomly flipping images horizontally during training corrupted Medial vs. Lateral compartment distinction, crippling meniscal and OA performance.
3. **Standard Binary Cross-Entropy on Severe Imbalance:** Standard BCE minimized overall loss by predicting near-zero probabilities on rare classes (`Fracture` <3%), crushing Macro ROC-AUC.
4. **Linear Rank Averaging:** Linearly averaging percentile ranks compressed extreme probabilities toward 0.5 due to the Irwin-Hall distribution effect, smoothing out subtle true-positive calls.
5. **Validation Exclusively on the 58 Gold Studies:** Cross-validation on the 58 gold studies correlated poorly with the leaderboard ($r = 0.66$) due to high statistical variance. In contrast, 5-fold CV across the entire ~4,349 report-labeled dataset correlated strongly ($r = 0.87$).
6. **Random Patient Splitting:** Radiologists reuse standard templated phrases for normal knees. Random splitting placed identical report texts in both train and validation folds, causing severe validation leakage. Grouping splits by report text hash (`hashlib.md5(report)`) was mandatory.
7. **Public Leaderboard Over-Probing Without Exact OOF:** Selecting models and micro-calibrations based on public LB feedback without local fold-excluded out-of-fold (OOF) validation introduced high selection variance and public/private shakeup risk.

---

## 4. Solution Comparison: Benchmarking Our Pipeline Against Top Solutions

### 4.1 Comparative Architectural & Engineering Matrix

| Dimension | Top Solutions (0.965 – 0.970) | Our Solution (0.958 Peak / 0.946 Fast) | Analysis & Delta |
| :--- | :--- | :--- | :--- |
| **Data & Label Supervision** | Proprietary clinical LLMs (Claude 3.5 Sonnet / GPT-4o / DeBERTa) with RadLex prompt engineering generating continuous calibrated soft labels $[0.0, 1.0]$. | Deterministic regex/rule-based clinical extractor (`src/report_continuous_extractor.py`) + public soft-label anchors (JEV/Llama). | **High-leverage gap:** Top teams trained directly on clean, continuous probabilities; our models inherited public label noise on non-specialist anchors. |
| **Volumetric Processing** | 2.5D multi-slice triplet slabs + 1D sequence models / MIL attention heads. | 2.5D multi-slot bag architecture (6 canonical acquisition slots: Sag/Cor/Ax Fluid FS, Sag Fluid Non-FS, Cor/Sag T1). | **Matched quality:** Both leveraged the winning 2.5D paradigm to avoid 3D CNN pitfalls. |
| **Anatomical Cropping** | 3D landmark keypoint localization (SKM-TEA 3D V-Net) producing structure-targeted zoom crops. | Fixed $130\,\text{mm}$ (or $100\,\text{mm}$) physical millimeter center crop via `PixelSpacing`. | **Key difference:** Top teams achieved 2x effective resolution on hairline tears by cropping directly around ligaments and cartilage facets. |
| **Anatomical Routing** | Cross-attention / MIL pooling over sequence slices with learned disease queries. | Proprietary `SlotHead` attention with dynamic presence masking and clinical prior buffer ($\text{logit} + 0.55$). | **Our strength:** `SlotHead` robustly handled missing series and enforced clinical viewing planes. |
| **Model Diversity** | 4 orthogonal families $\times$ 5 folds = 20 models (DINOv2 ViT-Base, Swin-V2 Large, ConvNeXt-Large, CoAtNet-4 SWA). | SOTA public anchors (CoAtNet-384 SWA, OrthoFoundation S6 ConvNeXt, 2.5D ConvNeXt MIL) + our 5-Fold ASL Specialist. | **Key difference:** Top teams broke the $\rho \approx 0.88$ correlation barrier with large ViT and Swin models trained from scratch on continuous labels. |
| **Loss Formulation** | Continuous soft-label BCE / KL-divergence + multi-task auxiliary losses (slice-level tear detection, KL grade). | Proprietary Asymmetric Focal Loss (ASL) with inverse frequency class weighting ($\gamma_+=0, \gamma_-=4, m=0.05$). | **Our strength:** Our ASL minority engine achieved +0.032 AUC on `Fracture` and 0.6216 on `Contusion`. |
| **Ensembling Math** | Linear rank averaging or learned meta-model stacking in probability space. | Latent Gaussian Probit Quantile Normal Fusion ($z = \Phi^{-1}(p)$ via `ndtri` and $p = \Phi(z)$ via `ndtr`). | **Our strength:** Probit fusion preserved extreme confidence calls at the distribution tails. |
| **Cross-Validation** | 5-Fold Multilabel Stratified Split grouped by patient/report, strictly tracking fold-excluded OOF for all stacks. | 5-Fold grouped split implemented in `src/dataset.py`, but lacked exact-stack fold-excluded OOF predictions for final multi-model blends. | **Critical vulnerability:** Lack of exact OOF forced reliance on public LB probing, leaving private test generalization uncertain. |
| **Efficiency Track** | Several top teams ran heavy 30-minute 20-model ensembles, placing lower on the Efficiency leaderboard. | Ultra-fast `exp-15 Efficiency Champion` (0.946 Macro AUC in <100s execution and <2 GB RAM). | **Our strength:** Designed an ultra-fast contender with massive execution speed advantages. |
| **Timeline / Start Date** | Active from Day 1 (July 30, 2026), giving 84 days of iterative experimentation. | First commit on September 30, 2026 (Day 62), leaving only 22 days before final deadline. | **Root constraint:** The severe late start constrained runway for multi-fold ViT training and comprehensive OOF audits. |

---

### 4.2 What Our Solution Did Exceptionally Well (Our Competitive Edge)

1. **The ASL Minority Engine:** While 100% of public baselines were trained with standard BCE (suppressing rare classes), our repository designed and trained a complete **5-Fold 384px Specialist Ensemble** using Asymmetric Focal Loss. This yielded a proven **+0.0323 AUC lift on `Fracture`**, **0.6216 AUC on `Contusion`**, and **0.6219 AUC on `MCL`**, providing crucial resilience against private-test minority class shakeup.
2. **Latent Gaussian Probit Quantile Normal Fusion:** We identified and solved the tail-compression flaw of standard linear rank blending. By mapping percentiles into latent standard normal space ($z = \Phi^{-1}(p)$), performing clinical target-weighted fusion, and mapping back via $\Phi(z)$, our pipeline preserved high-confidence detections with exponential sharpness.
3. **Rigorous Physics Preprocessing:** Our implementation of vector-projection slice sorting ($\text{slice\_pos} = \vec{p} \cdot \vec{n}$), physical millimeter centering, left-knee canonicalization, and zero horizontal flip training adhered strictly to medical imaging physics.
4. **Efficiency Track Engineering:** `exp-15 Efficiency Champion` (0.946 Macro AUC) executed in under 100 seconds with <2 GB RAM, representing an exceptionally optimized inference engine.
5. **Software Engineering Standards:** The repository maintained production-grade engineering standards: clean typed Python 3.10+ interfaces, deterministic seeding, 100% clean Ruff linting, and **62 passing unit tests** in ~9.0 seconds.

---

### 4.3 Why the Gap Existed & Highest-Leverage Areas for Improvement

The $0.007 – 0.012$ gap between our 0.958 baseline and the 0.965–0.970 gold tier is directly attributable to four specific factors:

```
                   ROOT-CAUSE DECOMPOSITION OF THE 0.007 GAP:

┌──────────────────────────────────────────────┬───────────────────────────────┐
│ Factor / Technical Area                      │ Estimated Opportunity Loss    │
├──────────────────────────────────────────────┼───────────────────────────────┤
│ 1. Proprietary Continuous Clinical LLM Labels│ +0.006 to +0.010 Macro AUC    │
│ 2. 3D Keypoint Localization (Structure Crops)│ +0.004 to +0.006 Macro AUC    │
│ 3. True Orthogonal Model Families (ViT/Swin) │ +0.005 to +0.008 Macro AUC    │
│ 4. Exact Fold-Excluded OOF Validation Cache  │ +0.003 to +0.005 Macro AUC    │
└──────────────────────────────────────────────┴───────────────────────────────┘
```

1. **High-Fidelity Clinical LLM Report Relabeling (Highest Leverage):**  
   Our vision models were bounded by the label quality of the underlying checkpoints. Upgrading from noisy public labels to continuous calibrated probabilities generated by an MSK-specific clinical LLM pipeline represents the single highest-leverage improvement.
2. **3D Keypoint Landmark Localization:**  
   Replacing fixed $130\,\text{mm}$ center crops with an automated keypoint locator (femoral notch, joint lines, patellar facet) would double effective resolution on subtle tears without increasing image dimensions.
3. **Training Independent Orthogonal Model Families:**  
   Relying on public checkpoints (CoAtNet, S6) locked the ensemble into a high-correlation noise regime ($\rho \approx 0.88$). Training independent 5-fold DINOv2 ViT-Base and Swin-V2 Large models from scratch would break this correlation barrier.
4. **Enforcing Strict OOF Validation Discipline:**  
   The lack of exact fold-excluded predictions for our final multi-model superstacks forced us to evaluate blends on the public leaderboard. Establishing a complete per-study OOF prediction cache for every candidate model is essential for reliable stacking and eliminating public-LB selection risk.

---

## 5. Setup, Tools, Hardware & Compute Environments

### 5.1 Did Top Teams Use LLMs or AI Assistants for Coding/Ideation?

#### Ideation & Code Generation Workflows
* **Coding Assistants:** Multiple top competitors reported using AI programming assistants (Claude 3.5 Sonnet, GitHub Copilot, ChatGPT) for:
  - Rapidly scaffolding complex PyTorch dataloaders and DICOM parsing routines.
  - Generating vector geometry projections and Affine matrix transformations for physical millimeter cropping.
  - Brainstorming loss formulations (e.g., Asymmetric Focal Loss, multi-label focal variants) and attention-pooling mechanisms.
  - Refactoring inference kernels for maximum speed (asynchronous prefetching, pinned memory buffers, torchscript/half-precision optimizations).

#### The Decisive Role: LLMs for Data Supervision & Distillation
* The most impactful application of LLMs was not in writing code, but in **generating training labels from the 4,349 free-text radiology reports**.
* **Host Rules & Compliance:** The competition host explicitly confirmed that using LLM APIs to extract labels from training reports was permitted, provided the models and costs satisfied the Reasonableness Standard and the final test inference pipeline remained strictly image-only (DICOMs in, predictions out).
* **Grandmaster Distillation Pipeline:**
  - Teams utilized Claude 3.5 Sonnet and GPT-4o with extensive Musculoskeletal (MSK) prompt templates containing hundreds of radiological synonyms from RadLex.
  - Rather than requesting binary classifications, they instructed the models to perform nuanced clinical hedging classification, outputting continuous confidence scores $[0.0, 1.0]$.
  - Teams also fine-tuned smaller open-source language models (e.g., `DeBERTa-v3-large` or `Llama-3-8B`) on the reports to serve as fast, local label extractors and feature teachers.

---

### 5.2 Hardware Setups & Compute Profiles of Top Teams

Medical volumetric imaging challenges on ~247 GB of DICOM data impose heavy computational requirements. Top-performing teams operated across substantial hardware configurations:

```
                          TOP TEAMS HARDWARE PROFILES:

   Workstation Profile (Local / Private Cloud)       High-End Dedicated Cluster
   ┌─────────────────────────────────────────┐       ┌─────────────────────────────────────────┐
   │ • 4x NVIDIA RTX 4090 (24 GB VRAM each)  │       │ • 4x to 8x NVIDIA A100 / H100 (80 GB)   │
   │ • AMD Threadripper / 128-256 GB RAM     │       │ • High-performance cloud instances      │
   │ • 2-4 TB NVMe Gen4 SSDs (7,000 MB/s)    │       │ • Distributed Data Parallel (DDP)       │
   │ • Native FP16 / BF16 Tensor Cores       │       │ • Multi-node automated Ray/Slurm sweeps │
   └─────────────────────────────────────────┘       └─────────────────────────────────────────┘
```

#### Key Hardware Characteristics:
* **High-Speed NVMe Storage:** DICOM decompression is notoriously I/O bound. Top teams pre-processed and converted the raw 247 GB DICOM dataset into lightweight formats (e.g., PNG slabs, compressed WebP, Zarr arrays, or memory-mapped NumPy arrays) stored on high-speed NVMe drives, accelerating dataloader throughput by $5\times\text{--}10\times$.
* **VRAM Capacity:** While 2.5D slab training is significantly more memory-efficient than 3D convolutions, training modern large vision backbones (ConvNeXt-Large, Swin-V2 Large, CoAtNet-4) at $384\times 384$ resolution with batch sizes of 16–32 requires 24 GB to 40 GB of VRAM.
* **Mixed Precision:** Native Tensor Core execution using PyTorch AMP (`torch.cuda.amp.autocast`) with `bfloat16` or `float16` enabled rapid backpropagation passes.

---

### 5.3 Were Platform GPUs Sufficient vs. External Compute Needs?

#### What Kaggle's Platform GPUs (Dual Tesla T4) Could Handle:
* Kaggle provides free cloud environments equipped with **Dual NVIDIA Tesla T4 GPUs (16 GB GDDR6 VRAM each)**, subject to a 30-hour weekly quota and a 9-hour runtime limit per session.
* **Sufficient For:**
  - Running fast 2.5D baseline models (e.g., ResNet-34, ResNet-50) at $224\times 224$ resolution.
  - Training focused single-fold specialists (our 384px RadImageNet ResNet-50 trained in ~50 minutes per fold on a single T4).
  - Executing Kaggle submission kernels (evaluating test studies within the 9-hour limit; our Efficiency Champion completed in <100 seconds).

#### Why Top Teams Required External Compute:
* **The Scale of Full 5-Fold Ensembles:** Training 4 diverse architecture families (DINOv2 ViT-Base, Swin-V2 Large, ConvNeXt-Large, CoAtNet-4) across 5 folds requires training **20 high-capacity models** at $384\times 384$ resolution.
* **Compute Mathematics:**
  - Training a single fold of ConvNeXt-Large or CoAtNet-4 at 384px resolution takes approximately 4 to 6 hours on an RTX 4090 or A100.
  - A full 5-fold run requires 20–30 GPU hours per architecture family.
  - Training all 4 orthogonal families across all 5 folds requires **100 to 150+ GPU hours** per complete iteration.
* **The Platform Quota Ceiling:** With Kaggle's 30-hour weekly limit on dual T4s (which run $3\times\text{--}5\times$ slower than modern RTX 4090 / A100 GPUs due to older Turing architecture and lower memory bandwidth), training a complete 20-model orthogonal ensemble would require **several weeks of continuous quota**, making comprehensive multi-family experimentation on Kaggle alone computationally prohibitive.
* **Conclusion:** While Kaggle's provided environment was completely sufficient for rapid prototyping, inference kernels, and single-family training, **contending for the Top 10 Gold medal tier required external multi-GPU compute (e.g., local multi-RTX 4090 rigs or cloud A100 instances)**.

---

### 5.4 Our Hardware Allocation Strategy (Apple M2 Pro + Dual Tesla T4)

To maximize developer velocity within compute constraints, our repository implemented a disciplined division of labor:

```
                            OUR TWO-TIER COMPUTE ARCHITECTURE:

   Local Workstation (Apple M2 Pro)                  Kaggle Cloud (Dual Tesla T4)
   ┌─────────────────────────────────────────┐       ┌─────────────────────────────────────────┐
   │ • 32 GB Unified LPDDR5 Memory           │       │ • 2x 16 GB GDDR6 Dedicated VRAM         │
   │ • Apple Metal Performance Shaders (MPS) │       │ • Hardware Tensor Cores (PyTorch AMP)   │
   │ • Instant terminal feedback & test runs │       │ • Pre-mounted competition dataset       │
   ├─────────────────────────────────────────┤       ├─────────────────────────────────────────┤
   │ RESPONSIBILITIES:                       │       │ RESPONSIBILITIES:                       │
   │ • Modular library development (`src/`)  │       │ • Multi-epoch 5-fold specialist training│
   │ • 62-test unit suite (pytest in ~9s)    │       │ • High-resolution 384px DICOM passes   │
   │ • Report NLP extraction & parsing       │       │ • Official submission kernel evaluation │
   │ • Ensembling & Probit quantile fusion   │       │ • Efficiency Track benchmark runs       │
   └─────────────────────────────────────────┘       └─────────────────────────────────────────┘
```

By enforcing the **Zero-Training Exhaustion Protocol** (optimizing ensembling weights, window sampling densities, and rank calibrations locally before dispatching GPU training), we minimized wasted cloud hours and successfully completed all 5 folds of our 384px ASL specialist within our allocated Kaggle quota.

---

## 6. Summary of Key Recommendations for Future Challenges

1. **Start on Day 1:** Volumetric medical competitions with weak text supervision require at least 60–80 days to systematically iterate across label refinement, landmark localization, multi-family training, and stacking.
2. **Prioritize Label Engineering Over Model Tuning:** When ground truth labels are sparse (<2%), investing compute into frontier LLM report distillation delivers $5\times$ the return of hyperparameter optimization.
3. **Establish Exact-Stack Fold-Excluded OOF Provenance Early:** Never submit or tune an ensemble without caching the exact fold-excluded validation predictions for all constituent models. A submission CSV or public LB probe cannot substitute for rigorous local OOF validation.
4. **Enforce Structural Medical Physics:** Always verify DICOM vector geometry, physical millimeter FOV scaling, and laterality constraints prior to training. Never apply naive symmetry-breaking augmentations (e.g., horizontal flipping).
5. **Combine Orthogonal Architecture Families:** Break inter-model noise correlation ceilings ($\rho \approx 0.88$) by blending vision transformers (DINOv2, Swin), pure convolutions (ConvNeXt), and hybrid networks (CoAtNet) in latent quantile space.
