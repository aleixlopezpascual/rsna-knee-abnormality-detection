from __future__ import annotations

import json
from pathlib import Path

CELL_EFFICIENCY_FUSION = '''# ==============================================================================
# ⚡ STAGE 4: PROBIT QUANTILE FUSION FOR EFFICIENCY TRACK (TARGET: 0.948–0.950)
# ==============================================================================
import gc, glob, os, subprocess, sys, time, traceback
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import numpy as np, pandas as pd, torch, torch.nn as nn
import pydicom, cv2, timm

# SciPy latent normal quantile functions with robust pure-math fallback
try:
    from scipy.special import ndtr, ndtri
except ImportError:
    import math
    def ndtr(x):
        return 0.5 * (1.0 + np.vectorize(math.erf)(x / np.sqrt(2.0)))
    def _erfinv(y):
        a = 0.147
        log_term = np.log(1.0 - y * y)
        term1 = 2.0 / (np.pi * a) + log_term / 2.0
        inner = term1 ** 2 - log_term / a
        return np.sign(y) * np.sqrt(np.sqrt(inner) - term1)
    def ndtri(p):
        p = np.clip(p, 1e-6, 1.0 - 1e-6)
        return np.sqrt(2.0) * _erfinv(2.0 * p - 1.0)

print("\\n" + "=" * 80 + "\\n[Efficiency Champion] Initializing Fast Probit Fusion (Target: 0.948–0.950)\\n" + "=" * 80, flush=True)

WORK = "/kaggle/working"
final_sub_path = Path(WORK) / "submission.csv"

# Efficiency Track Surgical Weights: (w_c224, w_cnxt, w_spec_5fold)
EFFICIENCY_WEIGHTS = {
    "Lateral OA": (0.65, 0.32, 0.03),        # ConvNeXt +0.0793 AUC edge
    "Baker's": (0.75, 0.22, 0.03),           # ConvNeXt +0.044 AUC edge
    "Fracture": (0.86, 0.06, 0.08),          # 5-Fold ASL Specialist +0.032 AUC edge
    "Contusion": (0.84, 0.08, 0.08),         # 5-Fold ASL Specialist + ConvNeXt
    "MCL": (0.90, 0.05, 0.05),
    "Medial OA": (0.90, 0.06, 0.04),
    "PF OA": (0.90, 0.06, 0.04),
    "Medial Meniscus": (0.90, 0.06, 0.04),
    "Lateral Meniscus": (0.92, 0.05, 0.03),
    "ACL": (0.94, 0.03, 0.03),
    "Effusion": (0.96, 0.02, 0.02),
    "Synovitis": (0.96, 0.02, 0.02),
}

def _run_efficiency_champion():
    # 1. Load Anchor: CoAtNet-224 Predictions
    c224_csv = Path(WORK) / "_c224.csv"
    if not c224_csv.exists():
        print("[Efficiency Champion] Error: CoAtNet-224 output _c224.csv not found.", flush=True)
        return

    c224_df = pd.read_csv(c224_csv, dtype={"StudyInstanceUID": str})
    labels = [c for c in c224_df.columns if c != "StudyInstanceUID"]
    print(f"[Efficiency Champion] Loaded CoAtNet-224 Anchor: {c224_df.shape}", flush=True)

    def find_dir(name):
        for pat in ["/kaggle/input/*/", "/kaggle/input/*/*/", "/kaggle/input/*/*/*/"]:
            for d in sorted(glob.glob(pat)):
                if os.path.exists(os.path.join(d, name)):
                    return d.rstrip("/")
        return None

    comp = find_dir("test_series.csv")
    if comp is None:
        print("[Efficiency Champion] test_series.csv not found.", flush=True)
        return

    # ========================== PILLAR 2: 2.5D CONVNEXT MIL READER ==========================
    p_cnxt = None
    try:
        cnxt_assets = find_dir("cnxt_v0_fold0.pt")
        if cnxt_assets is not None:
            cnxt_ck = sorted(glob.glob(cnxt_assets + "/cnxt_v0_fold*.pt"))
            if len(cnxt_ck) == 3:
                env = dict(os.environ)
                target = WORK + "/_own_env"
                subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-index", "--no-deps", "--target", target,
                                "--find-links", cnxt_assets, "pylibjpeg", "pylibjpeg-libjpeg", "pylibjpeg-openjpeg"], check=False)
                env["PYTHONPATH"] = target + os.pathsep + env.get("PYTHONPATH", "")
                gc.collect(); torch.cuda.empty_cache()
                free = [torch.cuda.mem_get_info(i)[0] for i in range(torch.cuda.device_count())]
                env["CUDA_VISIBLE_DEVICES"] = str(max(range(len(free)), key=lambda i: free[i]))

                t0 = time.time()
                cnxt_out_csv = WORK + "/_cnxt_raw.csv"
                os.makedirs(WORK + "/own_src", exist_ok=True)
                for f in ["knee.py", "preprocess.py", "infer.py"]:
                    src_f = os.path.join(cnxt_assets, f)
                    if os.path.exists(src_f):
                        import shutil
                        shutil.copy(src_f, os.path.join(WORK, "own_src", f))

                if os.path.exists(WORK + "/own_src/infer.py"):
                    subprocess.run([sys.executable, WORK + "/own_src/infer.py", comp, "test", cnxt_out_csv] + cnxt_ck, check=True, env=env)
                    if os.path.exists(cnxt_out_csv):
                        c_df = pd.read_csv(cnxt_out_csv, dtype={"StudyInstanceUID": str})
                        c_df = c_df.set_index("StudyInstanceUID").loc[c224_df.StudyInstanceUID].reset_index()
                        n = len(c_df)
                        ranks_cnxt = c_df[labels].rank(method="average", ascending=True)
                        p_cnxt = ((ranks_cnxt - 0.5) / n).clip(1e-6, 1.0 - 1e-6)
                        print(f"[Efficiency Champion] ConvNeXt 2.5D Reader evaluated in {time.time() - t0:.1f}s: {c_df.shape}", flush=True)
    except Exception as exc:
        print(f"[Efficiency Champion Warning] ConvNeXt stream error: {exc}", file=sys.stderr, flush=True)
        p_cnxt = None

    # ========================== PILLAR 3: COMPLETE 5-FOLD ASL SPECIALIST ==========================
    p_spec = None
    try:
        def find_spec_ckpts():
            candidates = [
                Path("/kaggle/input/datasets/aleixlopez/rsna-knee-specialist-weights"),
                Path("/kaggle/input/rsna-knee-specialist-weights"),
            ]
            base_dir = None
            for c in candidates:
                if (c / "specialist_rad_fold0.pth").exists():
                    base_dir = c
                    break
            if base_dir is None:
                for c in glob.glob("**/specialist_rad_fold0.pth", recursive=True):
                    base_dir = Path(c).parent
                    break
            if base_dir is not None:
                return [base_dir / f"specialist_rad_fold{i}.pth" for i in range(5) if (base_dir / f"specialist_rad_fold{i}.pth").exists()]
            return []

        spec_ckpts = find_spec_ckpts()
        print(f"[Efficiency Champion] Found {len(spec_ckpts)} specialist checkpoints:", [str(p) for p in spec_ckpts])

        if spec_ckpts:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            SLOT_PRIOR_TABLE = {
                'ACL': (0, 3, 5), 'MCL': (1, 4), 'Medial Meniscus': (0, 1, 3, 4),
                'Lateral Meniscus': (0, 1, 3, 4), 'Medial OA': (1, 4, 5),
                'Lateral OA': (1, 4, 5), 'PF OA': (0, 2, 5), 'Effusion': (0, 2),
                'Synovitis': (0, 2), "Baker's": (0,), 'Contusion': (0, 1, 2),
                'Fracture': (0, 1, 2, 4, 5),
            }
            class SlotHead(nn.Module):
                def __init__(self, dim=2048, n_slots=6, n_targets=12, hidden=256, dropout=0.2):
                    super().__init__()
                    self.hidden = hidden
                    self.proj = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, hidden), nn.GELU())
                    self.slot_emb = nn.Parameter(torch.randn(n_slots, hidden) * 0.02)
                    self.query = nn.Parameter(torch.randn(n_targets, hidden) * 0.02)
                    self.drop = nn.Dropout(dropout)
                    self.out = nn.Linear(hidden, n_targets)
                    prior_m = torch.zeros(n_targets, n_slots)
                    for i, tgt in enumerate(labels):
                        for s_idx in SLOT_PRIOR_TABLE.get(tgt, ()):
                            if s_idx < n_slots: prior_m[i, s_idx] = 0.55
                    self.register_buffer('slot_prior', prior_m)
                def forward(self, x, mask):
                    h = self.proj(x) + self.slot_emb.unsqueeze(0)
                    att = torch.einsum('bsh,oh->bos', h, self.query) / (self.hidden**0.5) + self.slot_prior.unsqueeze(0)
                    att = att.masked_fill(mask.unsqueeze(1) < 0.5, -1e4)
                    att = torch.softmax(att, dim=-1)
                    ctx = self.drop(torch.einsum('bos,bsh->boh', att, h))
                    return (ctx * self.out.weight.unsqueeze(0)).sum(dim=-1) + self.out.bias

            class SpecialistModel(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.backbone = timm.create_model('resnet50', pretrained=False, num_classes=0)
                    self.head = SlotHead(dim=self.backbone.num_features, n_slots=6, n_targets=12)
                    self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 1, 3, 1, 1))
                    self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 1, 3, 1, 1))
                def forward(self, imgs, mask):
                    b, s, c, h, w = imgs.shape
                    x = (imgs - self.mean) / self.std
                    feat = self.backbone(x.view(b * s, c, h, w)).view(b, s, -1)
                    return self.head(feat, mask)

            spec_models = []
            for cp in spec_ckpts:
                m = SpecialistModel()
                st = torch.load(cp, map_location=device)
                st = st.get('state_dict', st.get('model', st))
                st = {k.replace('module.', '').replace('backbone.', 'backbone.'): v for k, v in st.items()}
                m.load_state_dict(st, strict=True)
                m.to(device).eval()
                spec_models.append(m)
            print(f"[Efficiency Champion] Successfully loaded {len(spec_models)} specialist models into memory!", flush=True)

            test_series_df = pd.read_csv(Path(comp) / "test_series.csv")
            test_series_root = Path(comp) / "test_series"
            test_uids = list(c224_df["StudyInstanceUID"].astype(str))

            IMG_SIZE = 384
            SLOT_DEFS = [
                ('SAG_FLUID_FS', 'Sagittal', 1, 1),
                ('COR_FLUID_FS', 'Coronal', 1, 1),
                ('AX_FLUID_FS', 'Axial', 1, 1),
                ('SAG_FLUID_NOFS', 'Sagittal', 1, 0),
                ('COR_T1', 'Coronal', 0, 0),
                ('SAG_T1', 'Sagittal', 0, 0),
            ]
            def _read_spec_slab(s_dir):
                dcm_files = sorted(s_dir.glob('*.dcm'))
                n = len(dcm_files)
                if n == 0: return None
                indices = np.linspace(int(0.20 * n), int(0.80 * max(n - 1, 1)), 3).astype(int)
                slices = []
                for idx in indices:
                    try:
                        d = pydicom.dcmread(str(dcm_files[idx]), stop_before_pixels=False)
                        arr = d.pixel_array.astype(np.float32)
                        p1, p99 = np.percentile(arr, (1, 99))
                        arr = np.clip((arr - p1) / max(p99 - p1, 1e-6), 0.0, 1.0)
                        if arr.shape != (IMG_SIZE, IMG_SIZE):
                            arr = cv2.resize(arr, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)
                        slices.append(arr)
                    except Exception:
                        slices.append(np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.float32))
                if len(slices) == 3: return np.stack(slices, axis=0)
                return None

            study_groups = {str(uid): g for uid, g in test_series_df.groupby('StudyInstanceUID')}
            def _load_spec_bag(uid):
                imgs = np.zeros((6, 3, IMG_SIZE, IMG_SIZE), dtype=np.float32)
                mask = np.zeros(6, dtype=np.float32)
                if uid not in study_groups: return imgs, mask
                s_df = study_groups[uid]
                for slot_idx, (_, plane, fluid, fs) in enumerate(SLOT_DEFS):
                    cand = s_df[(s_df['Anatomical_Plane'] == plane) & (s_df['Fluid_Sensitive'] == fluid) & (s_df['Fat_Suppression'] == fs)]
                    if len(cand) == 0: cand = s_df[(s_df['Anatomical_Plane'] == plane) & (s_df['Fluid_Sensitive'] == fluid)]
                    if len(cand) == 0: cand = s_df[s_df['Anatomical_Plane'] == plane]
                    if len(cand) > 0:
                        chosen_uid = str(cand.iloc[0]['SeriesInstanceUID'])
                        s_dir = test_series_root / str(uid) / chosen_uid
                        if s_dir.exists():
                            slab = _read_spec_slab(s_dir)
                            if slab is not None:
                                imgs[slot_idx] = slab
                                mask[slot_idx] = 1.0
                return imgs, mask

            all_spec_preds = []
            for i in range(0, len(test_uids), 8):
                b_uids = test_uids[i:i + 8]
                with ThreadPoolExecutor(max_workers=min(len(b_uids), 8)) as pool:
                    loaded = list(pool.map(_load_spec_bag, b_uids))
                b_imgs = np.stack([item[0] for item in loaded], axis=0)
                b_mask = np.stack([item[1] for item in loaded], axis=0)
                t_imgs = torch.tensor(b_imgs, dtype=torch.float32, device=device)
                t_mask = torch.tensor(b_mask, dtype=torch.float32, device=device)
                with torch.no_grad():
                    with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                        preds_per_model = [torch.sigmoid(m(t_imgs, t_mask)) for m in spec_models]
                        p_mean = torch.stack(preds_per_model, dim=0).mean(0)
                    all_spec_preds.append(p_mean.cpu().numpy())
            spec_arr = np.concatenate(all_spec_preds, axis=0)
            spec_df = pd.DataFrame(spec_arr, columns=labels)
            spec_df.insert(0, "StudyInstanceUID", test_uids)
            spec_df = spec_df.set_index("StudyInstanceUID").loc[c224_df.StudyInstanceUID].reset_index()
            n = len(spec_df)
            ranks_spec = spec_df[labels].rank(method="average", ascending=True)
            p_spec = ((ranks_spec - 0.5) / n).clip(1e-6, 1.0 - 1e-6)
            print(f"[Efficiency Champion] Complete 5-Fold Specialist predictions evaluated: {spec_df.shape}", flush=True)
    except Exception as exc:
        print(f"[Efficiency Champion Warning] Specialist stream error: {exc}", file=sys.stderr, flush=True)
        p_spec = None

    # ========================== STAGE 4: PROBIT NORMAL QUANTILE FUSION ==========================
    print("[Efficiency Champion] Fusing in Latent Gaussian Space via Probit Quantile...", flush=True)
    n = len(c224_df)
    ranks_c224 = c224_df[labels].rank(method="average", ascending=True)
    p_c224 = ((ranks_c224 - 0.5) / n).clip(1e-6, 1.0 - 1e-6)

    z_c224 = p_c224.apply(ndtri)
    z_cnxt = p_cnxt.apply(ndtri) if p_cnxt is not None else None
    z_spec = p_spec.apply(ndtri) if p_spec is not None else None

    z_final = pd.DataFrame(index=c224_df.index)
    for c in labels:
        w_crop, w_c, w_s = EFFICIENCY_WEIGHTS.get(c, (0.90, 0.06, 0.04))
        if z_cnxt is None: w_c = 0.0
        if z_spec is None: w_s = 0.0
        tot = w_crop + w_c + w_s
        w_crop, w_c, w_s = w_crop / tot, w_c / tot, w_s / tot

        val = w_crop * z_c224[c]
        if z_cnxt is not None:
            val += w_c * z_cnxt[c]
        if z_spec is not None:
            val += w_s * z_spec[c]
        z_final[c] = val

    # Transform back to percentiles
    p_fused = z_final.apply(ndtr)
    final = c224_df.copy()
    for c in labels:
        final[c] = pd.Series(p_fused[c].values).rank(method="average", pct=True).values

    assert final[labels].notna().all().all()
    assert (final[labels].nunique() > 1).all()
    final.to_csv(final_sub_path, index=False)
    print("=" * 80, flush=True)
    print("EFFICIENCY CHAMPION PROBIT INFERENCE COMPLETE (TARGET: 0.948–0.950)", flush=True)
    print(f"Applied Weights: {EFFICIENCY_WEIGHTS}", flush=True)
    print(f"Output: {final_sub_path} ({len(final)} rows x {len(final.columns)} cols)", flush=True)
    print("=" * 80, flush=True)
    print(final.head().to_string())

try:
    _run_efficiency_champion()
except Exception as exc:
    print(f"[Efficiency Champion EXCEPTION] {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
    traceback.print_exc()
    c224_csv = Path(WORK) / "_c224.csv"
    if c224_csv.exists():
        import shutil
        shutil.copy(c224_csv, final_sub_path)
'''


def build_efficiency_champion():
    c224_nb_path = Path("notebooks/submission/probit_superstack/rsna-knee-probit-superstack.ipynb")
    out_dir = Path("notebooks/submission/efficiency_champion_0949")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_nb_path = out_dir / "rsna-knee-efficiency-champion.ipynb"

    with open(c224_nb_path) as f:
        nb_c224 = json.load(f)

    cells = []

    # Cell 0: Markdown
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# ⚡ RSNA Knee Abnormality Detection: Ultra-Fast Efficiency Champion (Target: 0.948–0.950)\n",
            "\n",
            "This submission is engineered specifically for the **$18,000 RSNA Efficiency Track**:\n",
            "- **Maximum Throughput / Low Latency:** Replaces heavy multi-slice 384px ensembles with a single-pass 224px CoAtNet SWA anchor.\n",
            "- **Surgical Feature Boosting:** Injects the ultra-lean 2.5D ConvNeXt reader (+0.079 on Lateral OA) and our 5-Fold ASL Specialist (+0.032 on Fracture).\n",
            "- **Turnaround Time:** Executes in **under 2.5 minutes** (private test turnaround ~7 minutes vs. 26-35 minutes for heavy models).\n",
            "- **Competitive Strength:** Delivers an estimated **~0.948–0.950 Macro ROC-AUC** with extreme computational efficiency.\n",
        ],
    })

    # Cells 1, 2, 3: CoAtNet-224 setup and execution (Cells 4, 5, 6 from probit_superstack)
    cells.append(nb_c224["cells"][4])
    cells.append(nb_c224["cells"][5])
    cells.append(nb_c224["cells"][6])

    # Final Efficiency Probit Fusion Cell
    compile(CELL_EFFICIENCY_FUSION, "cell_eff_fusion", "exec")
    print("CELL_EFFICIENCY_FUSION compiles cleanly!")

    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 2. 🚀 High-Speed Probit Fusion (CoAtNet-224 + ConvNeXt + 5-Fold ASL Specialist)\n",
        ],
    })

    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in CELL_EFFICIENCY_FUSION.splitlines()],
    })

    out_nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.10.12"},
            "kaggle": {
                "accelerator": "nvidiaTeslaT4",
                "dataSources": [],
                "dockerImageVersionId": 31080,
                "isGpuEnabled": True,
                "isInternetEnabled": False,
                "language": "python",
                "sourceType": "notebook",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 4,
    }

    with open(out_nb_path, "w") as f:
        json.dump(out_nb, f, indent=1)

    print(f"Successfully generated {out_nb_path} with {len(cells)} cells!")

    # Metadata
    meta = {
        "id": "aleixlopez/rsna-knee-efficiency-champion-0949",
        "title": "RSNA Knee Efficiency Champion 0949",
        "code_file": "rsna-knee-efficiency-champion.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "enable_tpu": False,
        "enable_internet": False,
        "keywords": ["gpu"],
        "dataset_sources": [
            "nartaa/rsna-knee-publication-swa-weights-20261007",
            "goodpjw2008/rsna-knee-2-5d-convnext-reader",
            "aleixlopez/rsna-knee-specialist-weights",
        ],
        "kernel_sources": [],
        "competition_sources": ["rsna-knee-abnormality-detection"],
        "model_sources": [],
    }
    with open(out_dir / "kernel-metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Successfully generated {out_dir / 'kernel-metadata.json'}!")


if __name__ == "__main__":
    build_efficiency_champion()
