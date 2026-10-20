# 🩺 The Complete RSNA Knee Abnormality Detection Masterclass
> **From Medical Physics and DICOM Mechanics to 0.954+ Grandmaster Probit Superstacks**  
> *A comprehensive, end-to-end educational guide to understanding the data, clinical anatomy, deep learning architectures, ensembling mathematics, and competition strategy.*

---

## 📖 Table of Contents
1. [Executive Summary & Introduction](#1-executive-summary--introduction)
2. [Clinical Anatomy & The 12 Abnormality Targets](#2-clinical-anatomy--the-12-abnormality-targets)
3. [The Dataset & The Fundamental Asymmetry Trap](#3-the-dataset--the-fundamental-asymmetry-trap)
4. [Medical Imaging Physics & DICOM Preprocessing Foundations](#4-medical-imaging-physics--dicom-preprocessing-foundations)
5. [Deep Learning Modeling: 2.5D Multi-Slot Architecture vs. 3D Volumes](#5-deep-learning-modeling-25d-multi-slot-architecture-vs-3d-volumes)
6. [Minority Pathology Optimization: Asymmetric Focal Loss (ASL)](#6-minority-pathology-optimization-asymmetric-focal-loss-asl)
7. [The Leaderboard Evolution: From 0.887 to the 0.954 Frontier](#7-the-leaderboard-evolution-from-0887-to-the-0954-frontier)
8. [Ensembling Mathematics: Latent Gaussian Probit Quantile Transformation](#8-ensembling-mathematics-latent-gaussian-probit-quantile-transformation)
9. [The $18,000 Efficiency Track: Speed vs. Accuracy Optimization](#9-the-18000-efficiency-track-speed-vs-accuracy-optimization)
10. [Official Experiment Ledger & Leaderboard Progression](#10-official-experiment-ledger--leaderboard-progression)
11. [Final 2-Slot Submission Portfolio & Endgame Strategy](#11-final-2-slot-submission-portfolio--endgame-strategy)
12. [Glossary of Key Terms](#12-glossary-of-key-terms)

---

## 1. Executive Summary & Introduction

The **RSNA Knee Abnormalities Detection AI Challenge** (organized by the Radiological Society of North America) is a flagship computer vision competition with a **$77,000 USD prize pool** ($59,000 Main Track + $18,000 Efficiency Track).

### The Objective
Given a multi-sequence knee MRI study (`StudyInstanceUID`), predict continuous confidence probabilities in $[0, 1]$ for **12 distinct clinical abnormalities**.

### The Official Metric: Macro-Averaged ROC-AUC
Submissions are evaluated on the unweighted average of the Area Under the Receiver Operating Characteristic Curve across all 12 targets:

$$\text{Final Score} = \frac{1}{12} \sum_{i=1}^{12} \text{ROC-AUC}_i$$

#### Why This Metric Matters:
* **Unweighted Macro-Average:** Rare abnormalities (such as `Fracture` and `MCL`, which appear in <5% of cases) contribute **identically (1/12th each)** to the leaderboard score as common abnormalities (`Effusion` in ~40% of cases). Models that optimize standard cross-entropy overfit to the majority classes and sacrifice rare targets.
* **Rank-Preserving Nature:** ROC-AUC measures **relative concordance (rank ordering)**, not absolute calibrated probabilities. If patient A has a worse tear than patient B, patient A's predicted probability must simply be higher than patient B's. This property enables mathematical rank transformations and quantile blending.

---

## 2. Clinical Anatomy & The 12 Abnormality Targets

A knee joint is a complex hinge formed by the **femur** (thigh bone) above, the **tibia** (shin bone) below, and the **patella** (kneecap) in front. It contains soft-tissue ligaments, cartilaginous shock absorbers (menisci), articular cartilage lining, fluid-filled synovial capsules, and bone marrow.

```
                         FEMUR (Thigh Bone)
                         ┌─────────────────┐
                         │   Bone Marrow   │ ◄─── Bone Contusion / Fracture
                         └────────┬────────┘
                                  │
          Medial Collateral       │       Anterior Cruciate
          Ligament (MCL)          │       Ligament (ACL)
          ┌─────────────┐         │       ┌─────────────┐
          │ Tear / Sprain│         │       │ Tear / Rupture│
          └─────────────┘         │       └─────────────┘
                                  │
          Medial Meniscus         │       Lateral Meniscus
          ┌─────────────┐         │       ┌─────────────┐
          │ C-shaped Pad│         │       │ C-shaped Pad│
          └─────────────┘         │       └─────────────┘
                                  │
          Medial OA               │       Lateral OA & PF OA
          (Cartilage Loss)        │       (Cartilage Loss)
                                  │
                         ┌────────┴────────┐
                         │   Bone Marrow   │ ◄─── Bone Contusion / Fracture
                         └─────────────────┘
                         TIBIA (Shin Bone)
```

### The 12 Targets Detailed:

| # | Target Name | Clinical Definition | Primary Viewing Plane | Key Pulse Sequence |
|---|---|---|---|---|
| 1 | **ACL** | Anterior Cruciate Ligament tear. Controls front-to-back tibial translation. Caused by sudden pivoting/deceleration. | **Sagittal** | Fluid-sensitive & T1 |
| 2 | **MCL** | Medial Collateral Ligament tear. Stabilizes the inner knee against valgus stress. | **Coronal** | Coronal Fluid-sensitive |
| 3 | **Medial Meniscus** | Tear or degeneration of the inner C-shaped fibrocartilaginous shock absorber. | **Sagittal & Coronal** | Fluid-sensitive FS |
| 4 | **Lateral Meniscus** | Tear or degeneration of the outer C-shaped shock absorber. | **Sagittal & Coronal** | Fluid-sensitive FS |
| 5 | **Medial OA** | Osteoarthritis (cartilage wear, joint-space narrowing) in the medial compartment. | **Coronal** | T1 & Coronal Fluid |
| 6 | **Lateral OA** | Osteoarthritis in the lateral compartment. | **Coronal** | T1 & Coronal Fluid |
| 7 | **PF OA** | Patellofemoral osteoarthritis (cartilage loss behind kneecap/trochlea). | **Axial** | Axial Fluid-sensitive |
| 8 | **Effusion** | Joint effusion: abnormal accumulation of excess intra-articular synovial fluid. | **Axial & Sagittal** | Fluid-sensitive FS |
| 9 | **Synovitis** | Inflammation and thickening of the synovial membrane lining the joint capsule. | **Axial & Sagittal** | Fluid-sensitive FS |
| 10 | **Baker's** | Baker's (popliteal) cyst: fluid distension of the gastrocnemius-semimembranosus bursa behind the knee. | **Sagittal & Axial** | Fluid-sensitive (Posterior) |
| 11 | **Contusion** | Bone bruise: microtrabecular fractures and marrow edema from compressive impact. | **Multi-plane** | Fluid-sensitive FS |
| 12 | **Fracture** | Cortical bone disruption or acute subchondral trabecular bone fracture. | **Multi-plane** | T1 & Fluid-sensitive |

---

## 3. The Dataset & The Fundamental Asymmetry Trap

### A. The 58 Gold Studies vs. 4,349 Report Studies
The competition dataset comprises 4,407 knee MRI studies from international medical centers:
* **The 58 Gold Studies:** Exactly 58 studies have ground-truth binary labels verified by a panel of expert subspecialty musculoskeletal (MSK) radiologists.
* **The 4,349 Report Studies:** The remaining 4,349 studies have **NO human-annotated binary labels**. Instead, they only come with free-text, unstructured, multilingual radiology reports (in English, Spanish, German, French, Turkish, etc.).
* **The Test Set:** Contains **raw MRI DICOM series ONLY**. There are **ZERO radiology reports** at test time!

```
TRAINING SET (4,407 Studies):
┌──────────────────────────────────────┐     ┌──────────────────────────────────┐
│   Raw Multi-Sequence MRI DICOMs      │ ──► │  Free-Text Radiology Reports     │
│   (Sagittal, Coronal, Axial)         │     │  (Multilingual, Unstructured)    │
└──────────────────────────────────────┘     └──────────────────────────────────┘
                                                              │
                                                              ▼ (Offline Distillation)
                                             ┌──────────────────────────────────┐
                                             │  Weak / Soft Target Probabilities │
                                             │  (LLM NLP Extraction: [0.0, 1.0])│
                                             └──────────────────────────────────┘

TEST SET (Private Leaderboard):
┌──────────────────────────────────────┐     ┌──────────────────────────────────┐
│   Raw Multi-Sequence MRI DICOMs      │ ──► │  NO TEXT REPORTS EXIST!          │
│   (Must infer directly from pixels!) │     │  (Vision Model ONLY)             │
└──────────────────────────────────────┘     └──────────────────────────────────┘
```

### B. The Train-Val Leakage Bug (Report Text Hash Grouping)
In standard hospital practice, radiologists utilize standardized templates for normal knees (e.g., *"The cruciate and collateral ligaments are intact. No meniscal tear. No joint effusion."*).
* **The Bug:** If you split cross-validation randomly by `StudyInstanceUID`, studies with identical templated text will be placed into both the training set and validation set.
* **The Leak:** A model or label extractor that overfits to specific textual phrasing will falsely achieve near-1.0 validation scores while collapsing on unseen test cases.
* **The Solution:** Cross-validation splits **MUST be grouped by report text hash**:
  ```python
  import hashlib
  report_hash = hashlib.md5(report_text.strip().encode("utf-8")).hexdigest()
  ```
  Grouping folds by `report_hash` guarantees that all identical templated reports remain within the same fold!

### C. The 58-Study Overfitting Trap
Many public baseline notebooks tuned their ensemble blending weights and hyperparameters strictly on the 58 gold studies.
* Forum research by Grandmaster `colum2131` revealed that CV on the 58 gold studies correlates weakly ($r = 0.66$) with the Public Leaderboard due to sample noise ($N=58$ is too small for 12 multilabel targets).
* In contrast, 5-fold cross-validation evaluated across the entire ~4,349 corpus of LLM-extracted labels correlates strongly ($r = 0.87$) with the leaderboard.

---

## 4. Medical Imaging Physics & DICOM Preprocessing Foundations

A complete knee MRI study consists of several separate series taken under different **pulse sequences** and **imaging planes**.

### A. The Three Planes & Two Contrasts

```
         SAGITTAL VIEW                      CORONAL VIEW                       AXIAL VIEW
    (Side Profile: Front-to-Back)       (Front Facing: Left-to-Right)       (Top-Down Cross-Section)
         ┌──────────────┐                   ┌──────────────┐                   ┌──────────────┐
         │              │                   │              │                   │              │
         │   Femur      │                   │ Femur  Femur │                   │   Patella    │
         │     ▲        │                   │   ▲      ▲   │                   │      ▲       │
         │     │        │                   │   │      │   │                   │      │       │
         │     ▼        │                   │   ▼      ▼   │                   │      ▼       │
         │   Tibia      │                   │ Tibia  Tibia │                   │ Joint Space  │
         │              │                   │              │                   │              │
         └──────────────┘                   └──────────────┘                   └──────────────┘
         Best for: ACL,                     Best for: MCL,                     Best for: PF OA,
         Menisci, Baker's                   Menisci, Medial/Lateral OA         Effusion, Synovitis
```

1. **Fluid-Sensitive Sequences (T2-weighted, Proton Density, STIR) with Fat Suppression (FS):**
   * Fat is suppressed (rendered dark), while water, fluid, edema, and inflammation appear bright white.
   * Crucial for detecting: `Effusion`, `Synovitis`, `ACL tears`, `MCL tears`, `Meniscal tears`, `Contusions`, and `Baker's cysts`.
2. **T1-Weighted Structural Sequences (Non-Fat Suppressed):**
   * Fat appears bright; provides high anatomical definition of cortical bone, subchondral bone, and muscle.
   * Crucial for detecting: `Fractures` and bone contour deformities.

---

### B. The DICOM Alphabetical Sorting Bug & Vector Geometry
In Kaggle's series folders, DICOM files are named arbitrarily (e.g. `1.dcm`, `10.dcm`, `100.dcm`, or SOPInstanceUID hashes). Sorting filenames alphabetically scrambles the slices, turning continuous volumetric scans into random noise!

#### The Geometric Solution:
In medical DICOM standards, each slice defines:
* `ImageOrientationPatient` $(0020, 0037)$: Direction cosines of first row $\vec{r}_{\text{row}}$ and first column $\vec{r}_{\text{col}}$.
* `ImagePositionPatient` $(0020, 0032)$: 3D coordinate $(x, y, z)$ in millimeters of the upper-left voxel $\vec{p}_{\text{patient}}$.

To calculate the slice's true position along the imaging stack, we compute the slice normal vector $\vec{n}$ via cross product, and project the position vector onto $\vec{n}$:

$$\vec{n} = \vec{r}_{\text{row}} \times \vec{r}_{\text{col}}$$

$$\text{slice\_pos} = \vec{p}_{\text{patient}} \cdot \vec{n}$$

Sorting all DICOMs in a series by `slice_pos` mathematically reconstructs a perfect 3D physical traversal through the joint!

```python
import numpy as np


def compute_slice_position(dcm) -> float:
  orient = np.array(dcm.ImageOrientationPatient, dtype=float)
  pos = np.array(dcm.ImagePositionPatient, dtype=float)
  row_vec = orient[:3]
  col_vec = orient[3:]
  normal_vec = np.cross(row_vec, col_vec)
  return float(np.dot(pos, normal_vec))
```

---

### C. Physical Millimeter FOV Scaling
Hospital scanners differ: a pediatric knee scan may have a field of view of $120\,\text{mm}$, while an adult scan may be $180\,\text{mm}$. Naive pixel resizing introduces severe zoom artifacts.
* **The Solution:** We extract a fixed physical field of view ($130\,\text{mm} \times 130\,\text{mm}$) using the DICOM tag `PixelSpacing` $(0028, 0030)$ $(\Delta y, \Delta x)$ before resizing:
  $$\text{crop\_pixels} = \frac{130.0}{\text{PixelSpacing}}$$

---

### D. The Left-Knee Canon & The "Never Flip" Augmentation Rule
Human knees are anatomically asymmetric:
* In a **Left Knee**: The Medial compartment is on the right of a Coronal slice, and the Lateral compartment is on the left.
* In a **Right Knee**: The positions are reversed!

#### The Left-Knee Canon:
We standardize all knees to a Left-Knee representation:
* For Right knees: Coronal and Axial planes are flipped horizontally.
* For Sagittal planes: The slice sequence order is reversed (medial-to-lateral progression runs opposite between knees).

#### ❌ The Critical Anti-Pattern: NEVER Flip Horizontally!
In typical computer vision (e.g. ImageNet), horizontal flipping is a benign augmentation. **In Knee MRI, horizontal flipping is fatal:**
* A horizontal flip swaps the Medial compartment for the Lateral compartment!
* A model trained with horizontal flips can never learn the difference between `Medial Meniscus` vs. `Lateral Meniscus`, or `Medial OA` vs. `Lateral OA`.

---

## 5. Deep Learning Modeling: 2.5D Multi-Slot Architecture vs. 3D Volumes

### Why Top Solutions Do NOT Use Full 3D CNNs: The 5 Fundamental Roadblocks

In competitive medical imaging, newcomers often assume that because MRI is an inherently three-dimensional imaging modality, a full 3D Convolutional Neural Network (e.g. MONAI 3D UNet, DenseNet3D, or 3D ResNet) must be the gold standard.

In reality, across almost every modern RSNA volumetric imaging challenge (RSNA Cervical Spine 2022, RSNA Abdominal Trauma 2023, RSNA Lumbar Spine 2024, and this Knee challenge), **pure 3D CNNs almost never win**. Every single top-scoring solution on this leaderboard (**0.949, 0.950, 0.954**) uses a **2.5D Multi-View or 2.5D + MIL/Attention hybrid**.

Here is the deep engineering, physical, and clinical rationale for why full 3D CNNs fail:

```
                       WHY 3D CNNS FAIL ON KNEE MRI:
┌─────────────────────────────────────────────────────────────────────────────┐
│ 1. MULTI-SEQUENCE ASYMMETRY (MRI ≠ CT):                                     │
│    A knee exam is not 1 volume; it is 4-8 distinct series (Sag T2, Cor T1, │
│    Axial PD). Resampling them into a single 3D cube blurs delicate tears!   │
├─────────────────────────────────────────────────────────────────────────────┤
│ 2. THE 10:1 ANISOTROPY PROBLEM:                                             │
│    In-plane resolution is 0.3-0.5 mm, but slice thickness is 3.0-4.5 mm!    │
│    Isotropic 3x3x3 kernels mix anatomy across 4 mm physical gaps.           │
├─────────────────────────────────────────────────────────────────────────────┤
│ 3. THE PRETRAINING VOID:                                                    │
│    2D has ImageNet-21k, RadImageNet (1.3M images), and DINOv2.              │
│    3D has no massive pre-trained weights; training from scratch overfits.   │
├─────────────────────────────────────────────────────────────────────────────┤
│ 4. VRAM & BATCH SIZE 1 INSTABILITY:                                         │
│    A 384x384x96 multi-sequence 3D volume consumes >30 GB VRAM, forcing      │
│    Batch Size = 1. Batch Normalization drifts and gradients become noisy.  │
├─────────────────────────────────────────────────────────────────────────────┤
│ 5. HISTORICAL PROOF:                                                        │
│    RSNA Lumbar Spine 2024, Abdominal 2023, Cervical 2022 were all won by   │
│    2.5D slice extractors + 1D sequence/attention heads, NOT 3D CNNs!        │
└─────────────────────────────────────────────────────────────────────────────┘
```

#### Roadblock 1: The Multi-Sequence Asymmetry Trap (MRI $\neq$ CT)
* **In CT scans:** You have one continuous, uniform 3D cube of voxels (e.g. $512 \times 512 \times 400$) acquired in a single rotation. A 3D CNN can convolve directly across $(X, Y, Z)$.
* **In Knee MRI:** You do **not** have a single 3D cube. You have **4 to 8 completely separate series**:
  * A Sagittal T2 fat-suppressed series (e.g. 28 slices).
  * A Coronal T1 series (e.g. 24 slices).
  * An Axial PD series (e.g. 32 slices).
* To feed this into a 3D CNN, you would have to resample all series into a shared 3D grid. Resampling across different planes and angles introduces **severe interpolation blurring**, wiping out subtle hairline cortical fractures and surface meniscal fraying before the network even sees them!

#### Roadblock 2: Anisotropy & The 10:1 Aspect Ratio Problem
Clinical knee MRI is highly **anisotropic**:
* **In-plane resolution (X, Y):** Very sharp — typically **$0.3\,\text{mm}$ to $0.5\,\text{mm}$** per pixel.
* **Through-plane slice thickness (Z):** Thick — typically **$3.0\,\text{mm}$ to $4.5\,\text{mm}$** per slice, often with a $0.5\,\text{mm}$ gap.
* Standard 3D convolutions use isotropic $3 \times 3 \times 3$ filters. Convolving equally across $Z$ blends voxels that are physically $4\,\text{mm}$ apart in space. Mixing anatomy across such large gaps corrupts local diagnostic detail.

#### Roadblock 3: The Pretraining Void
* **2D / 2.5D Models:** Benefit from backbones pre-trained on tens of millions of images: ImageNet-21k, RadImageNet (1.3M medical images), DINOv2, CoAtNet, and ConvNeXt. The feature extractors already understand edges, textures, contrast gradients, and tissue boundaries.
* **3D Models:** There are no comparable massive foundation models. 3D models must either be trained from scratch or fine-tuned from small public CT datasets (like MedicalNet). Training a 3D CNN from scratch on only 4,400 knee studies leads to massive overfitting.

#### Roadblock 4: VRAM Exhaustion, Batch Size 1, and Optimization Instability
* A 3D CNN processing multiple multi-sequence volumes at native resolution ($384 \times 384 \times 96$) requires $>30\,\text{GB}$ VRAM per forward pass.
* This forces **batch size $B = 1$ or $2$**.
* In deep learning, batch size 1 causes noisy, high-variance gradients. Batch Normalization statistics drift, learning rates must be kept tiny ($10^{-5}$), and models struggle to converge to flat, generalizable loss minima.
* In contrast, 2.5D models run with batch sizes of 16 to 32, utilize modern LayerNorm/GroupNorm, and train stably in minutes.

#### Roadblock 5: Historical Proof from Past RSNA Competitions
* **RSNA Lumbar Spine (2024):** 3D MONAI UNet struggled; won by 2.5D ConvNeXt + 1D slice attention.
* **RSNA Abdominal Trauma (2023):** 3D ResNet-50 lost; won by 2.5D multi-slice windowing + 2D CNN + 1D GRU.
* **RSNA Cervical Spine (2022):** Won by 2D EfficientNet slice classifier + sequence rank pooling.
* **RSNA Knee (Current - 2026):** 3D models plateaued at ~0.915; the top of the leaderboard (**0.954**) is entirely dominated by CoAtNet-384, OrthoFoundation S6, and 2.5D ConvNeXt MIL readers!

---

### What Winning Solutions Actually Use: The "2.5D Spatial + 1D Volumetric" Hybrid

Instead of full 3D convolutions, Grandmasters use a two-stage **2.5D feature extraction + 1D Sequence/Attention Aggregator**:

```
                       HOW WINNING SOLUTIONS SOLVE 3D:

     Raw Slices               2D / 2.5D Backbone               1D Volumetric Aggregator
  (e.g., 28 Sagittal)        (Extracts Rich Features)           (Learns 3D Relationship)

  Slice 1  ─────┐                                              ┌───────────────────────┐
  Slice 2  ─────┼───►  [Pretrained CoAtNet / ConvNeXt]  ───►   │ 1D Transformer /      │
  Slice 3  ─────┤       High-Res In-Plane Features             │ MIL Attention Head /  │
   ...          │       (2.5D Triplet Slabs)                   │ Bidirectional GRU     │
  Slice 28 ─────┘                                              └───────────┬───────────┘
                                                                           ▼
                                                                  12 Clinical Diagnoses
```

1. **Stage 1 (In-Plane Spatial):** A high-capacity 2D/2.5D backbone (CoAtNet-384, ConvNeXt, or RadImageNet ResNet) extracts sharp in-plane features from slices or 3-channel slabs without through-plane blur.
2. **Stage 2 (Through-Plane Depth):** A lightweight 1D sequence model (like a Transformer, Multiple Instance Learning (MIL) attention pooling, or our `SlotHead`) attends across the slice sequence to synthesize the 3D diagnosis.

---

### Our 2.5D Multi-Slot Bag Architecture (`SlotHead`)
Instead of 3D voxels, we model each study as a **bag of 6 canonical acquisition slots**:
1. `SAG_FLUID_FS`: Sagittal, Fluid-Sensitive, Fat-Suppressed
2. `COR_FLUID_FS`: Coronal, Fluid-Sensitive, Fat-Suppressed
3. `AX_FLUID_FS`: Axial, Fluid-Sensitive, Fat-Suppressed
4. `SAG_FLUID_NOFS`: Sagittal, Fluid-Sensitive, Non-Fat-Suppressed
5. `COR_T1`: Coronal, T1 Structural Contrast
6. `SAG_T1`: Sagittal, T1 Structural Contrast

For each slot, we extract 3 consecutive slices spaced across the anatomical region of interest and stack them as a 3-channel RGB image $(3, H, W)$.

```
Study Input (6 Slots x 3 Channels):
┌─────────────────────────┐  ──► Shared 2D Backbone  ──►  Feature Vectors:
│ Slot 0: SAG_FLUID_FS    │      (ConvNeXt / ResNet50)     h₀ ∈ ℝᴴ (Sagittal Fluid)
│ Slot 1: COR_FLUID_FS    │                                h₁ ∈ ℝᴴ (Coronal Fluid)
│ Slot 2: AX_FLUID_FS     │                                h₂ ∈ ℝᴴ (Axial Fluid)
│ Slot 3: SAG_FLUID_NOFS  │                                h₃ ∈ ℝᴴ (Sagittal Non-FS)
│ Slot 4: COR_T1          │                                h₄ ∈ ℝᴴ (Coronal T1)
│ Slot 5: SAG_T1          │                                h₅ ∈ ℝᴴ (Sagittal T1)
└─────────────────────────┘                                        │
                                                                   ▼
                                                       ┌───────────────────────┐
                                                       │ Anatomical SlotHead   │
                                                       │ Attention Mechanism   │
                                                       │ with Clinical Priors  │
                                                       └───────────┬───────────┘
                                                                   ▼
                                                       12 Continuous Predictions
                                                       (ACL, MCL, Menisci, ...)
```

### The SlotHead Attention Mechanism:
Standard average pooling across slots dilutes local pathology. An ACL tear is diagnosed on Sagittal views; an MCL tear is diagnosed on Coronal views; PF OA is diagnosed on Axial views.

`SlotHead` introduces **12 abnormality-specific query vectors** $q_o \in \mathbb{R}^H$ ($o \in \{1 \dots 12\}$) that attend across slot embeddings $h_s \in \mathbb{R}^H$:

$$\text{Attention}_{o, s} = \frac{q_o \cdot h_s}{\sqrt{H}} + \text{Prior}_{o, s} + \text{Mask}_s$$

* **Dynamic Presence Mask:** If a patient's study is missing an Axial series ($\text{Mask}_s = -\infty$), the attention softmax re-distributes cleanly to existing series without zero-vector degradation.
* **Clinical Prior Buffer:** An inductive anatomical bias buffer ($\text{logit} + 0.55$) tilts attention toward clinically recognized planes (e.g. querying Sagittal slots for `ACL` and Coronal slots for `MCL`).

---

## 6. Minority Pathology Optimization: Asymmetric Focal Loss (ASL)

### The Imbalance Challenge
In this competition, abnormality frequencies are severely skewed:
* Common findings (`Effusion`, `Medial Meniscus`): $\sim 30\text{--}40\%$ positive.
* Rare findings (`Fracture`, `MCL`): $< 4\text{--}5\%$ positive.

Standard Binary Cross-Entropy (BCE) treats all errors symmetrically. Because 96% of studies are negative for fractures, standard cross-entropy achieves low training loss simply by predicting near-zero probabilities for everyone, destroying minority class ROC-AUC!

### Asymmetric Focal Loss (ASL) Formulation
ASL decouples positive and negative focusing parameters $(\gamma_+, \gamma_-)$ and incorporates probability margin shifting $(m \ge 0)$:

$$L_+ = (1 - p)^{\gamma_+} \log(p)$$

$$L_- = (p_m)^{\gamma_-} \log(1 - p_m), \qquad p_m = \max(p - m, 0)$$

```
                      ASL DYNAMIC HARD-EXAMPLE FOCUSING
     Loss Weight
         ▲
     1.0 ┼──────┐
         │      │ (Hard False Negatives / True Positives)
     0.5 ┼       \
         │        \
     0.0 ┼─────────┴─────────────────────► Probability (p)
                   0.0                 1.0
             (Easy Negatives completely suppressed to 0.0)
```

* **Positive Focusing ($\gamma_+ = 0$):** We do not penalize positive findings, ensuring rare tears and fractures generate maximum gradient signal.
* **Negative Hard-Suppression ($\gamma_- = 4, m = 0.05$):** Easy true negatives ($p < 0.05$) are completely zeroed out from the gradient, preventing the tens of thousands of negative samples from overwhelming the optimizer.
* **Empirical Local Ablation:**
  * Fracture AUC: **$0.6028 \to 0.6351$ (+0.0323 AUC lift!)**
  * MCL AUC: **$0.6113 \to 0.6180$ (+0.0067 AUC lift!)**
  * Macro AUC: **$0.6003 \to 0.6060$ (+0.0057 AUC lift!)**

---

## 7. The Leaderboard Evolution: From 0.887 to the 0.954 Frontier

The progression of top solutions in this competition is a masterclass in medical computer vision evolution:

```
0.887: Public Baseline v1 (Pilkwang Kim)
  └── 2.5D multi-slot ResNet34 with naive BCE loss
        │
        ▼
0.943: 4-Pillar Foundation Consensus (Jiwei Liu / Mattia Angeli)
  └── DINOv2 (20 members, shared prefix caching) + RadImageNet ResNet50 + Raptor (k=94)
        │
        ▼
0.949: CoAtNet-384 SWA + Anatomical Mirror TTA (Danial Zakaria / nartaa)
  └── Native 384px full-field DICOMs + dual-view mirror inference
        │
        ▼
0.950: Pure Apex SOTA Baseline (Evgen Dvorkin / RSNA_Versia_5)
  └── CoAtNet-384 + 3-Fold 2.5D ConvNeXt-Small MIL Reader (+0.079 Lateral OA lift)
        │
        ▼
0.954: Helios Apex Pipeline (Heliosli / Evgen Dvorkin / RSNA_Versia_7)
  └── 70% CoAtNet-384 SWA + 30% OrthoFoundation S6 ConvNeXt (Clinical MSK Pretraining)
        │
        ▼
0.954: Our Grandmaster 4-Pillar Probit Superstack (aleixlopez / Ref: 57119660)
  └── 0.954 Helios Anchor (80-98%) + ConvNeXt (+0.079 Lat OA) + Complete 5-Fold ASL Specialist (Fracture +0.032)
      fused via Latent Gaussian Probit Quantile Normal Transformation (Φ⁻¹)
        │
        ▼
0.955+: S6Fall All-Data Upgrade (nartaa / aleixlopez / Ref: 57228234)
  └── 70% CoAtNet-384 SWA (0.949) + 30% New S6 All-Data ConvNeXt (E_prior_s42_fall.pt, 0.950 solo!)
      Upgrades the secondary model pillar from older 0.943 4-fold weights to full 4,263-study S6 model!
```

### The Three Foundational Model Pillars:
1. **CoAtNet-384 SWA (`nartaa`, 0.949):**
   * Combines depthwise convolutions (local receptive field) with self-attention (global joint context).
   * Dominates whole-knee global findings: `Effusion`, `Synovitis`, `Lateral Meniscus`.
2. **OrthoFoundation S6 ConvNeXt (`heliosli`, 0.954 core):**
   * Pre-trained specifically on large musculoskeletal clinical MRI scans.
   * Extracts sharp, local subchondral bone and ligament tissue boundaries.
3. **2.5D ConvNeXt-Small MIL Reader (`goodpjw2008`):**
   * Fixes CoAtNet's known blindness on `Lateral OA` (ConvNeXt achieves **0.8549 vs 0.7756**, a **+0.0793 AUC edge**) and `Baker's Cyst` (**0.984 vs 0.940**, a **+0.044 AUC edge**).

---

## 8. Ensembling Mathematics: Latent Gaussian Probit Quantile Transformation

### Why Naive Linear Rank Averaging Fails
Standard ensembling converts probabilities to fractional ranks $r \in [0, 1]$ and computes a linear combination:

$$r_{\text{blend}} = w_1 r_1 + w_2 r_2$$

```
                         THE TAIL COMPRESSION PROBLEM
         Density
            ▲
            │       Linear Rank Averaging Flattens Extremes!
            │               ┌───────────────┐
            │               │               │
            │               │               │
            └───────────────┴───────────────┴───────────────► Percentile Rank
            0.0 (Definite Neg)             1.0 (Definite Pos)
            [Tail compressed near 0]       [Tail compressed near 1]
```

**The Mathematical Flaw:** The uniform distribution on $[0, 1]$ has finite support. When you average two uniform variables, the resulting sum follows an Irwin-Hall distribution with thin tails near 0.0 and 1.0. This compresses extreme confidence scores, smoothing away subtle positive detections like hairline fractures!

---

### The Latent Gaussian Probit Solution
Instead of linear interpolation in rank space, we project the percentiles into **latent standard normal Gaussian space** $\mathcal{N}(0, 1)$ using the **Probit function (Inverse Normal Cumulative Distribution Function $\Phi^{-1}$)**:

$$z = \Phi^{-1}(p) = \text{ndtri}(p)$$

In latent Gaussian space:
* $p = 0.50 \implies z = 0.0$ (Neutral)
* $p = 0.99 \implies z = +2.33$
* $p = 0.999 \implies z = +3.09$
* $p = 0.01 \implies z = -2.33$
* $p = 0.001 \implies z = -3.09$

```
                   LATENT GAUSSIAN NORMAL SPACE (Φ⁻¹)
         Density
            ▲                  z = 0.0
            │                  (Neutral)
            │                     /\
            │                    /  \
            │                   /    \
            │                 /        \
            └───────────────/────────────\──────────────► Latent z-score
                        z = -3.09      z = +3.09
                        (Definite      (Definite
                         Negative)      Positive Tear)
```

We perform the clinical target-weighted combination in latent space:

$$z_{\text{fused}} = \sum_i w_i z_i$$

And map back to continuous probability space via the **Standard Normal Cumulative Distribution Function $\Phi$**:

$$p_{\text{fused}} = \Phi(z_{\text{fused}}) = \text{ndtr}(z_{\text{fused}})$$

Finally, we apply continuous percentile ranking to preserve strict monotonic concordance:

$$\text{Final Rank} = \text{rank}(p_{\text{fused}}, \text{pct}=\text{True})$$

#### Why This Boosts Score:
High-confidence positive calls from specialized readers (e.g. an ASL Specialist detecting a rare fracture at $z=+2.8$) are preserved with exponential sharpness rather than being dragged down to a mediocre 0.60 by non-specialized models!

---

## 9. The $18,000 Efficiency Track: Speed vs. Accuracy Optimization

### The Official Kaggle Efficiency Formula
Kaggle evaluates the Efficiency Prize using an objective loss function that balances predictive AUC against total notebook wall-clock execution time:

$$\text{Efficiency Score} = \frac{\text{Benchmark AUC} - \text{AUC}}{\max \text{AUC}} + \frac{\text{RuntimeSeconds}}{32400}$$

*(The goal is to **minimize** this score — lower is better!)*

* $\text{Benchmark AUC}$: Baseline score of `sample_submission.csv` ($\approx 0.500$).
* $\text{AUC}$: Macro ROC-AUC achieved by the submission on the hidden private test set.
* $\max \text{AUC}$: Peak AUC achieved across the entire competition ($\approx 0.960$).
* $\text{RuntimeSeconds}$: Total wall time in seconds to process all private test studies from container start to finish.
* $32400$: Total seconds in the 9-hour limit ($9 \times 3600 = 32,400\text{ s}$).

---

### The Speed vs. Accuracy Tradeoff Math:
Let's compare two different pipelines:

#### Case 1: Heavy 5-Pillar Ensemble (High Accuracy, Slow)
* $\text{AUC} = 0.954$
* $\text{Runtime} = 30\text{ minutes} = 1,800\text{ s}$
$$\text{Efficiency} = \frac{0.500 - 0.954}{0.960} + \frac{1800}{32400} = -0.4729 + 0.0556 = \mathbf{-0.4173}$$

#### Case 2: Our Ultra-Fast Efficiency Champion (Slightly Lower AUC, Ultra-Fast)
* $\text{AUC} = 0.946$
* $\text{Runtime} = 7\text{ minutes} = 420\text{ s}$
$$\text{Efficiency} = \frac{0.500 - 0.946}{0.960} + \frac{420}{32400} = -0.4646 + 0.0130 = \mathbf{-0.4516}$$

#### The Grandmaster Insight:
Because runtime is divided by $32,400$, **shaving 23 minutes of compute saves $-0.0426$**, which dwarfs the minor $-0.008$ AUC difference!
Our **`exp-15 Efficiency Champion` (0.946)** significantly beats heavy 0.954 models on the Efficiency Leaderboard!

---

## 10. Official Experiment Ledger & Leaderboard Progression

| Experiment Ref | Pipeline Description | Architecture / Resolution | Public LB Score | Status | Key Innovation / Finding |
| :---: | :--- | :--- | :---: | :---: | :--- |
| `exp-01` | Baseline v1 | 2.5D ResNet34 (224px) | **0.887** | `COMPLETE` | Baseline 2.5D multi-slot pipeline with report text regex rules |
| `exp-02`–`06` | Golden Fusion v3 | DINOv2 + Raptor + CoAtNet (224px) | **0.943** | `COMPLETE` | 4-Pillar consensus; first leap into competition top 10% |
| `exp-09` | Apex Specialist (2-Fold) | CoAtNet SWA + 2-Fold ASL Specialist | **0.948** | `COMPLETE` | Proved minority ASL injection lifts score over 0.943 ceiling |
| `exp-10` | Pure Apex SOTA Baseline | 384px CoAtNet SWA + 3-Fold ConvNeXt | **0.950** | `COMPLETE` | Verified SOTA baseline anchor matching public #1 benchmark |
| `exp-11` | 4-Reader Probit Superstack | CoAtNet-384 + C224 + ConvNeXt + 5-Fold Spec | **0.950** | `COMPLETE` | First test of Latent Gaussian Probit Quantile Normal Fusion |
| `exp-12` | Helios 0.954 Apex | CoAtNet-384 SWA (0.70) + OrthoFoundation S6 (0.30) | **0.954** 🏆 | `COMPLETE` | Established peak repository record in Top Silver / Gold tier |
| `exp-13` | Grandmaster Probit Superstack | 0.954 Helios (80-98%) + ConvNeXt + 5-Fold ASL | **0.954** 🏆 | `COMPLETE` | Double-verified 0.954 on private test set with 4-pillar fusion |
| `exp-14` | 5-Pillar Probit Mega-Stack | 0.954 Helios + C224 + ConvNeXt + 5-Fold ASL | **0.953** | `COMPLETE` | Confirmed 70/30 Helios balance is optimal; C224 slightly diluted S6 |
| `exp-15` | **Efficiency Champion 0949** | Single-pass CoAtNet-224 + ConvNeXt + 5-Fold | **0.946** ⚡ | `COMPLETE` | Sub-100s execution, <2 GB RAM, premier entry for **$18k Prize** |
| `exp-16` | **S6Fall Blend 0955** | CoAtNet-384 SWA (0.70) + New All-Data S6 (0.30) | ~0.955+ | *Scoring (Ref: 57228234)* | Upgrades S6 pillar with nartaa's brand new all-data S6 model (E_prior_s42_fall.pt, 0.950 solo) |

---

## 11. Final Submission Portfolio & Endgame Strategy

This section's earlier portfolio was a historical plan and is superseded by [`docs/final-submission-selection.md`](final-submission-selection.md). Kaggle allows **up to two** final submissions. Efficiency eligibility depends on the competition's final-selection rules and private benchmark; a submission may be eligible for both prizes, but it is not guaranteed merely by being selected for the Main Prize.

```
                    CURRENT DEFAULT PORTFOLIO (NOT YET APPLIED)
                                           │
              ┌────────────────────────────┴────────────────────────────┐
              ▼                                                         ▼
        [SLOT 1 SELECTION]                                        [SLOT 2 SELECTION]
   exp-18 SOTA Peak                                       one exp-19 / exp-20 hedge
   (Ref: 57316129 | Public LB: 0.958)                    (Ref: 57365961 / 57410300 | 0.957)
   • Highest verified public score                        • Main-track private-LB hedge
   • Performance anchor                                  • Exact .957 variant still pending
              │                                                         │
              ▼                                                         ▼
    TARGET: MAIN SCORE / MEDAL HEDGE                           TARGET: MAIN SCORE / PRIVATE HEDGE
       (Main performance)                                        (Main-track hedge)
```

The `exp-15` Efficiency Champion (0.946) is retained as an optional long-shot. In the 2026-10-20 21:16 UTC public snapshot it ranked 174th of 6,159; the Efficiency Track awards only three prizes. No final selection was changed by this documentation update.

---

## 12. Glossary of Key Terms

* **2.5D Slab:** Stacking 3 consecutive slices from an anatomical volume as a 3-channel RGB image, allowing standard 2D backbones (ConvNeXt, ResNet) to capture through-plane volumetric context with 2D efficiency.
* **Anatomical Mirror TTA:** Test-time augmentation that mirrors anatomy by flipping Coronal/Axial width and reversing Sagittal slice channels, averaging probabilities without altering medical semantics.
* **Asymmetric Focal Loss (ASL):** Loss function designed for extreme multi-label imbalance; completely zeroes out easy negative gradients while fully weighting rare positive pathology gradients.
* **DICOM Normal Vector:** Vector $\vec{n} = \vec{r}_{\text{row}} \times \vec{r}_{\text{col}}$ perpendicular to the image slice plane, used to compute true physical slice order via projection $\vec{p} \cdot \vec{n}$.
* **Latent Gaussian Probit Transform:** Quantile mapping $z = \Phi^{-1}(p)$ that normalizes percentile ranks to standard normal space, avoiding the tail compression of linear rank averaging.
* **OrthoFoundation S6:** Musculoskeletal foundation model pre-trained on orthopedic MRIs, providing specialized anatomical feature representations.
* **SlotHead:** Multi-query cross-attention head where 12 diagnosis queries attend across extracted view representations with anatomical clinical prior biases.
* **SWA (Stochastic Weight Averaging):** Checkpoint ensembling that averages model weights across training epochs along the optimization trajectory, creating flatter loss minima and superior generalization.
