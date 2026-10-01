# Compute Architecture & Hardware Strategy

**Repository:** RSNA Knee Abnormalities Detection (`rsna-knee-abnormality-detection`)  
**Lead ML Engineering Guidelines**

---

## 1. Local Hardware Profile

- **Processor:** Apple M2 Pro
- **Unified Memory (RAM):** 32 GB LPDDR5
- **Local Storage:** 480 GB available NVMe SSD
- **Hardware Acceleration:** Apple Metal Performance Shaders (MPS)
  - `torch.backends.mps.is_available() == True`
  - Unified memory architecture eliminates CPU-to-GPU PCIe transfer latency.

---

## 2. Local vs. Cloud Compute Comparison

| Dimension | Local Workstation (Apple M2 Pro) | Kaggle Cloud Environment (Dual Tesla T4) |
| :--- | :--- | :--- |
| **Dataset Access** | Must download ~247 GB raw DICOMs (4–8h over broadband) | **Instantaneous** pre-mounted network storage (`/kaggle/input/`) |
| **VRAM / Memory** | Up to ~24 GB dynamically allocated unified RAM | 2x 16 GB dedicated GDDR6 VRAM |
| **FP16 / BF16 Tensor Cores** | Emulated / MPS float16 | **Hardware Tensor Cores** ($3\times\text{--}5\times$ faster backprop) |
| **Iterative Debugging** | **Instant** terminal feedback, pytest in 4s | Queued batch jobs, deferred log streaming |
| **Best Used For** | Fast prototyping, unit tests, ensembling, NLP parsing | Heavy 5-fold training, large volumetric inferences |

---

## 3. Division of Labor & Compute Discipline

To optimize both developer velocity and compute efficiency:

1. **Local Machine Responsibilities:**
   - Writing and verifying all modular codebase logic (`src/`).
   - Maintaining our 23-test unit test suite (`pytest tests/`).
   - Parsing and extracting weak labels from the 4,407 radiology reports (`src/report_reader.py`).
   - Testing geometric slice ordering, laterality canonicalization, and data transformations.
   - Ensembling, percentile rank calibration, and weight optimization experiments (`src/ensemble.py`).
   - Building and packaging kernel bundles for Kaggle deployment.

2. **Kaggle Cloud Responsibilities:**
   - Full-scale multi-epoch model training on the 247 GB raw DICOM training set.
   - Rerunning inference kernels against the ~1,300+ hidden test studies.
   - Utilizing dual Tesla T4 GPUs with PyTorch AMP (`torch.cuda.amp.autocast`).

---

## 4. The Zero-Training Exhaustion Mandate

Before spending GPU hours training models from scratch, we must follow this mandatory Grandmaster discipline:
1. **Never train blind:** Every training run must have a clear, measurable hypothesis targeting specific underperforming abnormalities.
2. **Exhaust zero-training options first:** Tune ensembling weights, volumetric window sampling densities, and rank calibrations on existing pretrained checkpoints before training new weights.
3. **Evidence before assertion:** Validate local CV gains on a single fold before running full 5-fold training.
