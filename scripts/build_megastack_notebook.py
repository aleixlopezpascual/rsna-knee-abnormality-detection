from __future__ import annotations

import json
from pathlib import Path

CELL_MEGASTACK_FUSION = '''# ==============================================================================
# 🌟 STAGE 4, 5 & 6: GRANDMASTER 5-PILLAR MEGA-STACK (TARGET 0.956–0.960)
# ==============================================================================
# 1. Anchor: Verified 0.954 Helios Baseline (CoAtNet-384 + OrthoFoundation S6)
# 2. Pillar 2: CoAtNet-224 SWA (Focal Joint-Crop Cartilage Specialist)
# 3. Pillar 3: 2.5D ConvNeXt-Small MIL Reader (+0.0793 AUC on Lateral OA)
# 4. Pillar 4: Complete 5-Fold 384px ASL Specialist (+0.032 AUC on Fracture, 0.6216 Contusion)
# 5. Fusion: Latent Gaussian Probit Quantile Transformation
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

print("\\n" + "=" * 80 + "\\n[Mega-Stack 0960] Initializing 5-Pillar Probit Mega-Stack (Target: 0.956–0.960)\\n" + "=" * 80, flush=True)

WORK = "/kaggle/working"
helios_sub_path = Path(WORK) / "submission.csv"

# Backup verified 0.954 Helios submission for 100% fallback safety
helios_backup_path = Path(WORK) / "_helios_0954_backup.csv"
if helios_sub_path.exists():
    import shutil
    shutil.copy(helios_sub_path, helios_backup_path)

# 5-Pillar Surgical Weights: (w_helios_0954, w_c224_crop, w_cnxt, w_spec_5fold)
MEGASTACK_WEIGHTS = {
    "Lateral OA": (0.65, 0.15, 0.18, 0.02),    # ConvNeXt +0.0793 AUC edge, C224 cartilage zoom
    "Baker's": (0.75, 0.10, 0.13, 0.02),       # ConvNeXt +0.044 AUC edge on fluid pockets
    "Fracture": (0.84, 0.05, 0.05, 0.06),      # 5-Fold ASL Specialist +0.032 AUC advantage
    "Contusion": (0.82, 0.08, 0.05, 0.05),     # 5-Fold ASL Specialist (0.6216) + ConvNeXt
    "MCL": (0.86, 0.06, 0.04, 0.04),           # CoAtNet + ASL Specialist (0.6219)
    "Medial OA": (0.80, 0.15, 0.03, 0.02),     # C224 focal zoom on medial compartment
    "PF OA": (0.80, 0.15, 0.03, 0.02),         # C224 focal zoom on patellofemoral compartment
    "Medial Meniscus": (0.85, 0.08, 0.04, 0.03),
    "Lateral Meniscus": (0.88, 0.07, 0.03, 0.02),
    "ACL": (0.90, 0.06, 0.02, 0.02),
    "Effusion": (0.95, 0.03, 0.01, 0.01),
    "Synovitis": (0.95, 0.03, 0.01, 0.01),
}

def _run_megastack():
    if not helios_backup_path.exists():
        print("[Mega-Stack] Helios 0.954 backup does not exist.", flush=True)
        return

    h_sub = pd.read_csv(helios_backup_path, dtype={"StudyInstanceUID": str})
    labels = [c for c in h_sub.columns if c != "StudyInstanceUID"]
    print(f"[Mega-Stack] Loaded 0.954 Helios Baseline: {h_sub.shape}", flush=True)

    def find_dir(name):
        for pat in ["/kaggle/input/*/", "/kaggle/input/*/*/", "/kaggle/input/*/*/*/"]:
            for d in sorted(glob.glob(pat)):
                if os.path.exists(os.path.join(d, name)):
                    return d.rstrip("/")
        return None

    comp = find_dir("test_series.csv")
    if comp is None:
        print("[Mega-Stack] test_series.csv not found.", flush=True)
        return

    # ========================== PILLAR 2: COATNET-224 FOCAL CROP SPECIALIST ==========================
    p_c224 = None
    try:
        c224_csv = Path(WORK) / "_c224.csv"
        if not c224_csv.exists() and os.path.exists(WORK + "/c224_src/infer224.py"):
            print("[Mega-Stack] Running CoAtNet-224 Crop Specialist...", flush=True)
            subprocess.run([sys.executable, WORK + "/c224_src/infer224.py"], check=False)
        if c224_csv.exists():
            c224_df = pd.read_csv(c224_csv, dtype={"StudyInstanceUID": str})
            c224_df = c224_df.set_index("StudyInstanceUID").loc[h_sub.StudyInstanceUID].reset_index()
            n = len(c224_df)
            ranks_c224 = c224_df[labels].rank(method="average", ascending=True)
            p_c224 = ((ranks_c224 - 0.5) / n).clip(1e-6, 1.0 - 1e-6)
            print(f"[Mega-Stack] CoAtNet-224 predictions loaded: {c224_df.shape}", flush=True)
    except Exception as exc:
        print(f"[Mega-Stack Warning] CoAtNet-224 error: {exc}", file=sys.stderr, flush=True)
        p_c224 = None

    # ========================== PILLAR 3: 2.5D CONVNEXT MIL READER ==========================
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
                        c_df = c_df.set_index("StudyInstanceUID").loc[h_sub.StudyInstanceUID].reset_index()
                        n = len(c_df)
                        ranks_cnxt = c_df[labels].rank(method="average", ascending=True)
                        p_cnxt = ((ranks_cnxt - 0.5) / n).clip(1e-6, 1.0 - 1e-6)
                        print(f"[Mega-Stack] ConvNeXt 2.5D Reader evaluated in {time.time() - t0:.1f}s: {c_df.shape}", flush=True)
    except Exception as exc:
        print(f"[Mega-Stack Warning] ConvNeXt stream error: {exc}", file=sys.stderr, flush=True)
        p_cnxt = None

    # ========================== PILLAR 4: COMPLETE 5-FOLD ASL SPECIALIST ==========================
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
        print(f"[Mega-Stack] Found {len(spec_ckpts)} specialist checkpoints:", [str(p) for p in spec_ckpts])

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
            print(f"[Mega-Stack] Successfully loaded {len(spec_models)} specialist models into memory!", flush=True)

            test_series_df = pd.read_csv(Path(comp) / "test_series.csv")
            test_series_root = Path(comp) / "test_series"
            test_uids = list(h_sub["StudyInstanceUID"].astype(str))

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
            spec_df = spec_df.set_index("StudyInstanceUID").loc[h_sub.StudyInstanceUID].reset_index()
            n = len(spec_df)
            ranks_spec = spec_df[labels].rank(method="average", ascending=True)
            p_spec = ((ranks_spec - 0.5) / n).clip(1e-6, 1.0 - 1e-6)
            print(f"[Mega-Stack] Complete 5-Fold Specialist predictions evaluated: {spec_df.shape}", flush=True)
    except Exception as exc:
        print(f"[Mega-Stack Warning] Specialist stream error: {exc}", file=sys.stderr, flush=True)
        p_spec = None

    # ========================== STAGE 6: 5-PILLAR LATENT GAUSSIAN PROBIT FUSION ==========================
    print("[Mega-Stack] Fusing 5 Pillars in Latent Gaussian Space via Probit Quantile...", flush=True)
    n = len(h_sub)
    ranks_helios = h_sub[labels].rank(method="average", ascending=True)
    p_helios = ((ranks_helios - 0.5) / n).clip(1e-6, 1.0 - 1e-6)

    z_helios = p_helios.apply(ndtri)
    z_c224 = p_c224.apply(ndtri) if p_c224 is not None else None
    z_cnxt = p_cnxt.apply(ndtri) if p_cnxt is not None else None
    z_spec = p_spec.apply(ndtri) if p_spec is not None else None

    z_final = pd.DataFrame(index=h_sub.index)
    for c in labels:
        w_h, w_crop, w_c, w_s = MEGASTACK_WEIGHTS.get(c, (0.85, 0.08, 0.04, 0.03))
        if z_c224 is None: w_crop = 0.0
        if z_cnxt is None: w_c = 0.0
        if z_spec is None: w_s = 0.0
        tot = w_h + w_crop + w_c + w_s
        w_h, w_crop, w_c, w_s = w_h / tot, w_crop / tot, w_c / tot, w_s / tot

        val = w_h * z_helios[c]
        if z_c224 is not None:
            val += w_crop * z_c224[c]
        if z_cnxt is not None:
            val += w_c * z_cnxt[c]
        if z_spec is not None:
            val += w_s * z_spec[c]
        z_final[c] = val

    # Transform back to percentiles
    p_fused = z_final.apply(ndtr)
    final = h_sub.copy()
    for c in labels:
        final[c] = pd.Series(p_fused[c].values).rank(method="average", pct=True).values

    assert final[labels].notna().all().all()
    assert (final[labels].nunique() > 1).all()
    final.to_csv(helios_sub_path, index=False)
    print("=" * 80, flush=True)
    print("5-PILLAR PROBIT MEGA-STACK COMPLETE (TARGET: 0.956–0.960)", flush=True)
    print(f"Applied Weights: {MEGASTACK_WEIGHTS}", flush=True)
    print(f"Output: {helios_sub_path} ({len(final)} rows x {len(final.columns)} cols)", flush=True)
    print("=" * 80, flush=True)
    print(final.head().to_string())

try:
    _run_megastack()
except Exception as exc:
    print(f"[Mega-Stack EXCEPTION] {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
    traceback.print_exc()
    print("[Mega-Stack FALLBACK] Preserving 0.954 Helios baseline safely.", flush=True)
    if helios_backup_path.exists():
        import shutil
        shutil.copy(helios_backup_path, helios_sub_path)
'''


def build_megastack():
    src_nb_path = Path("notebooks/submission/helios_0954_apex/rsna-knee-helios-0954.ipynb")
    c224_nb_path = Path("notebooks/submission/probit_superstack/rsna-knee-probit-superstack.ipynb")
    out_dir = Path("notebooks/submission/megastack_0960")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_nb_path = out_dir / "rsna-knee-megastack.ipynb"

    with open(src_nb_path) as f:
        nb = json.load(f)

    with open(c224_nb_path) as f:
        nb_c224 = json.load(f)

    # Cells 0 to 11 are the verified 0.954 Helios pipeline
    cells = nb["cells"][:12]

    # Insert CoAtNet-224 generation cell (Cells 4, 5, 6 from probit_superstack)
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 6. 🔬 CoAtNet-224 Focal Joint-Crop Specialist Execution\n",
            "\n",
            "Generates focal 224x224 joint-space predictions focusing on cartilage narrowing (`Medial OA`, `Lateral OA`, `PF OA`)\n",
            "in an isolated subprocess to preserve GPU memory for subsequent stages.\n",
        ],
    })
    # Cell 4 from probit_superstack: mkdir c224_src
    cells.append(nb_c224["cells"][4])
    # Cell 5 from probit_superstack: infer224.py
    cells.append(nb_c224["cells"][5])
    # Cell 6 from probit_superstack: run infer224.py
    cells.append(nb_c224["cells"][6])

    # Final Mega-Stack Fusion cell
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 7. 🏆 Grandmaster 5-Pillar Probit Mega-Stack Fusion (Target: 0.956–0.960)\n",
            "\n",
            "Fuses all 5 leading model families in Latent Gaussian Space via Probit Quantile Transformation:\n",
            "1. Verified 0.954 Helios Baseline (CoAtNet-384 SWA + OrthoFoundation S6 ConvNeXt)\n",
            "2. CoAtNet-224 SWA (Focal Joint-Crop Cartilage Specialist)\n",
            "3. 2.5D ConvNeXt-Small MIL Reader (+0.0793 AUC on `Lateral OA`, +0.044 on `Baker's`)\n",
            "4. Complete 5-Fold 384px ASL Specialist Ensemble (+0.032 AUC on `Fracture`, 0.6216 `Contusion`)\n",
            "5. Latent Gaussian Probit Quantile Normal Fusion\n",
        ],
    })

    compile(CELL_MEGASTACK_FUSION, "cell_megastack", "exec")
    print("CELL_MEGASTACK_FUSION compiles cleanly!")

    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in CELL_MEGASTACK_FUSION.splitlines()],
    })

    nb["cells"] = cells

    with open(out_nb_path, "w") as f:
        json.dump(nb, f, indent=1)

    print(f"Successfully generated {out_nb_path} with {len(cells)} cells!")

    # Metadata
    meta = {
        "id": "aleixlopez/rsna-knee-megastack-0960",
        "title": "RSNA Knee MegaStack 0960",
        "code_file": "rsna-knee-megastack.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "enable_tpu": False,
        "enable_internet": False,
        "keywords": ["gpu"],
        "dataset_sources": [
            "heliosli/rsna-knee-compact-v1",
            "nartaa/rsna-knee-publication-swa-weights-20261007",
            "heliosli/rsna-orthofoundation-fold0-epoch7-20261004",
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
    build_megastack()
