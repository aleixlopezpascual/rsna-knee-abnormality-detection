# Top Public Notebooks: RSNA Knee Abnormalities Detection

Comprehensive analysis and catalog of the highest-scoring public notebooks on Kaggle, reverse-engineered from source code, runtime traces, attached weights, and leaderboard rankings.

---

## 🏆 Top Notebooks Leaderboard Ranking

| Public LB | Title | Author | Kernel Ref | Core Innovation |
| :---: | :--- | :--- | :--- | :--- |
| **0.950** | `RSNA_Versia_5` (v5) | Evgen Dvorkin | `evgendvorkin/rsna-versia-5` | **SOTA Apex Fusion:** SOTA 0.949 CoAtNet SWA + 3-Fold 2.5D ConvNeXt MIL Reader with target-calibrated rank fusion |
| **0.949** | `rsna-knee-0-949-coatnet-stack-blend` | Pilkwang Kim | `goodpjw2008/rsna-knee-0-949-coatnet-stack-blend-lb-0-949` | 60% nartaa 0.949 CoAtNet + 40% 0.943 community stack rank blend |
| **0.949** | `RSNA Knee 0949 Anatomical Mirror` | Danial Zakaria | `nartaa/rsna-knee-0949-anatomical-mirror` | **Highest scoring single model (0.949):** 384px CoAtNet RMLP-2 SWA with Anatomical Mirror TTA |
| **0.946** | `rsna-knee-d4-public0946` | yamadan96 | `yamadan96/rsna-knee-d4-public0946` | 4-way CoAtNet consensus + Raptor k=94 + DINOv2 |
| **0.946** | `rsna-knee-d4-blend` | pjmathematician | `pjmathematician/rsna-knee-d4-blend` | SOTA D4 blend with hardened runtime contract and asset checks |
| **0.943** | `RSNA Knee Fast 2xT4 Inference` | Jiwei Liu | `jiweiliu/rsna-knee-fast-2xt4-inference` | Dual-GPU T4 parallelization and asynchronous CPU prefetch |
| **0.942** | `Bend the Knee to Speedy Raptors` | Mattia Angeli | `mattiaangeli/bend-the-knee-to-speedy-raptors-the-original` | Foundational Speedy Raptors architecture (DINO shared prefix + Raptor) |
| **0.940** | `Hybrid Raptor + Meniscus Specialist` | renta.k | `renta0426/rsna-knee-hybrid-raptor-meniscus-specialist` | Discovered Lateral Meniscus 1.00 CoAtNet routing specialization |
| **0.937** | `Head and Shoulders, Knees and Toes` | prvsiyan | `prvsiyan/head-and-shoulders-knees-and-toes` | Expanded Raptor to 4 slicing arms + global CoAtNet fusion |
| **0.887** | `RSNA Knee baseline v1` | Pilkwang Kim | `pilkwang/rsna-knee-baseline-v1` | Foundational 2.5D multi-slot pipeline, geometric ordering, NLP report reader |

---

## 🧬 Architectural Evolution: 0.887 → 0.940 → 0.943 → 0.946 → 0.949 → 0.950

```
0.887: Pure DINOv2 small 20-member ensemble
  └── + Raptor volumetric slicing (4 arms) + CoAtNet
        └── 0.937 (prvsiyan)
              └── + Isolated Lateral Meniscus 1.00 routing
                    └── 0.940 (renta0426)
                          └── + Shared DINO prefix + 2xT4 GPU concurrency
                                └── 0.942 (mattiaangeli)
                                      └── + 4-Way CoAtNet Consensus (Global96 + RepairV1) + k_eval=94
                                            └── 0.943 (jiweiliu)
                                                  └── + Rank-space D4 consensus blend
                                                        └── 0.946 (yamadan96 / pjmathematician)
                                                              └── + 384px CoAtNet SWA + Anatomical Mirror TTA
                                                                    └── 0.949 (nartaa)
                                                                          └── + 3-Fold 2.5D ConvNeXt MIL Reader
                                                                                └── 0.950 (evgendvorkin: RSNA_Versia_5)
```

---

## 🏛️ The 4 Pillars of the 0.945+ Public Solutions

```
                          ┌───────────────────────────────────────┐
                          │   Multi-Sequence Knee MRI DICOMs      │
                          └───────────────────┬───────────────────┘
                                              │
         ┌───────────────────┬────────────────┴───────────────────┬───────────────────┐
         │                   │                                    │                   │
         ▼                   ▼                                    ▼                   ▼
┌──────────────────┐┌──────────────────┐                 ┌──────────────────┐┌──────────────────┐
│ Pillar 1: DINOv2 ││  Pillar 2: Rad   │                 │ Pillar 3: Raptor ││ Pillar 4: CoAtNet│
│  Shared Prefix   ││ ImageNet ResNet50│                 │   4 Slicing Arms ││  4-Way Consensus │
│   (20 Members)   ││  (E10/E11/E13)   │                 │   (k_eval = 94)  ││ (ResGated/Global)│
└────────┬─────────┘└────────┬─────────┘                 └────────┬─────────┘└────────┬─────────┘
         │                   │                                    │                   │
         └─────────┬─────────┘                                    └─────────┬─────────┘
                   ▼                                                        ▼
         [Transformer Stack]                                      [Convolutional Stack]
                   │                                                        │
                   └──────────────────────────┬─────────────────────────────┘
                                              ▼
                             ┌──────────────────────────────────┐
                             │  Target-Specific Routing Matrix  │
                             │  (e.g., Lat. Meniscus = 1.0 CoAt)│
                             └────────────────┬─────────────────┘
                                              ▼
                             ┌──────────────────────────────────┐
                             │   Percentile Rank Submission     │
                             └──────────────────────────────────┘
```

### Pillar 1: DINOv2 with Shared Prefix Acceleration
- **Backbone:** DINOv2-small (`metaresearch/dinov2/PyTorch/small/1`), 768 feature dimension.
- **Shared Prefix Trick:** The first 6 transformer layers are **byte-identical** across all 20 members (`SHARED_DINO_PREFIX_SHA256`). Top kernels evaluate the first 6 blocks **once** per study, caching the activation tensor and evaluating only the differing final 6 blocks + head for each of the 20 members. This saves ~80% of computation time.
- **A5 Attention Folds:** 5-fold cross-attention models from `mattiaangeli/knee-mri-fold-weights` blended at `a5_w = 0.52`.

### Pillar 2: Medical Pretraining with RadImageNet (ResNet-50)
- **Source:** `marwanmath/resnet-50-radimagenet-marwan` (`ResNet50.pt`).
- **Advantage:** Pre-training on 1.3 million medical images (CT/MRI/X-ray) substantially outperforms generic ImageNet for MRI bone/cartilage features.
- **Inference Engine:** Dual persistent replicas on `cuda:0` and `cuda:1` with asynchronous CPU slice prefetching (`_RadLayoutPrefetch`), combined with an 88-feature calibrator on gated findings. Mixed into transformer stack at `rad_alpha = 0.55`.

### Pillar 3: Raptor Volumetric Slicing Views (4 Arms)
Raptor extracts 4 complementary volumetric views across the joint:
1. `maxspan-v5` (weight `0.60`): Wide coverage from femur to tibia.
2. `native384dense-v10` (weight `0.10`): High-resolution $384 \times 384$ dense slicing.
3. `maxspan-v5-reverse` (weight `0.10`): Inverted slice order for traversal invariance.
4. `native384-v8` (weight `0.20`): Native resolution structural window.
- **Window Density (`k_eval = 94`):** Increased window sampling from 62/42 to **94 windows per study**, providing dense sampling across the joint with no model changes.

### Pillar 4: CoAtNet Multi-Scale Hybrid Consensus
Combines convolutional local feature capture with transformer global self-attention across 4 specialized CoAtNet variants:
1. `resgated_top3` (residual gating)
2. `d4_depthzone_swa3` (depth-zone aware stochastic weight averaging)
3. `global96_top3` (global context)
4. `repairv1_top3`

### Pillar 5: ConvNeXt 2.5D Multi-Window Reader (`goodpjw2008/rsna-knee-2-5d-convnext-reader`)
- **Origin & Discovery:** Reverse-engineering of top 0.946 public kernels revealed that the jump from 0.943 to 0.946 relies on blending a private 2.5D ConvNeXt reader at ~10% weight.
- **Pretrained Weights:** Discovered public dataset `goodpjw2008/rsna-knee-2-5d-convnext-reader` containing pre-trained 3-fold weights (`cnxt_v0_fold0.pt`, `cnxt_v0_fold1.pt`, `cnxt_v0_fold2.pt`).
- **Architecture (`KneeNet`):** ConvNeXt-Tiny 2D backbone extracting features from 3-slice windows, embedded with anatomical slot IDs and normalized z-axis slice positions, processed via a 2-layer `TransformerEncoder` and 12-target attention pooling head.
- **Ensemble Role:** Provides completely orthogonal, non-overlapping inductive bias to the DINOv2 and CoAtNet consensus.

---

## 🎯 The Proven Target-Specific Anatomical Routing Matrix

Top notebooks avoid uniform blending. They apply a clinical routing matrix where individual targets are routed to the model family that clinically excels at that finding:

```python
# The proven 0.943 - 0.946 routing configuration
coatnet_w = {
    "__default__": 0.60,        # 60% CoAtNet / 40% Transformer for general findings
    "Lateral Meniscus": 1.00,   # 100% CoAtNet (CoAtNet dramatically dominates here)
    "Medial Meniscus": 0.80,    # 80% CoAtNet
    "ACL": 0.75,                # 75% CoAtNet
    "Lateral OA": 0.75,         # 75% CoAtNet
    "Fracture": 0.75,           # 75% CoAtNet
}
```

- **Lateral Meniscus Specialization:** Routing `Lateral Meniscus` exclusively to CoAtNet (`1.00`) provided a direct $+0.003$ public LB boost because CNN local receptive fields detect radial and horizontal tears far more cleanly than attention tokens.
- **Percentile Rank Fusion:** Blending is strictly performed in rank space (`rank(method="average", pct=True)`), neutralizing score distribution differences between model families.
