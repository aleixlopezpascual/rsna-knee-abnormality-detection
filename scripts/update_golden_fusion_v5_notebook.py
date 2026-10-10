from __future__ import annotations

import json
from pathlib import Path

CELL_31_V5_CODE = '''# ============================ GOLDEN FUSION v5: 6-PILLAR HYPER-ENSEMBLE ============================
# Fuses Foundation Anchor (0.943) + 3-Fold ConvNeXt 2.5D Reader (0.946 key) + ASL Rare-Finding Specialist (384px)
import gc
import glob
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import pandas as pd
import pydicom
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

print("\\n" + "=" * 80 + "\\n[golden-fusion-v5] Initializing 6-Pillar Hyper-Ensemble\\n" + "=" * 80, flush=True)

_TARGET_COLS = [
    "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA",
    "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"
]

# 3-Way Target-Specific Fusion Weights: (Foundation, ConvNeXt, Specialist)
_FUSION_WEIGHTS_3WAY = {
    "Fracture": (0.50, 0.20, 0.30),          # Specialist ASL sensitivity boost
    "MCL": (0.55, 0.25, 0.20),
    "Lateral Meniscus": (0.60, 0.20, 0.20),  # CoAtNet dominates, ConvNeXt + Specialist add key margin
    "Contusion": (0.55, 0.25, 0.20),         # Specialist (0.6182 AUC) + ConvNeXt
    "__default__": (0.65, 0.25, 0.10),        # 25% ConvNeXt injection across all findings
}

_sub_path = Path("/kaggle/working/submission.csv")

def _find_convnext_weights():
    candidates = [
        Path("/kaggle/input/datasets/goodpjw2008/rsna-knee-2-5d-convnext-reader"),
        Path("/kaggle/input/rsna-knee-2-5d-convnext-reader"),
    ]
    base_dir = None
    for c in candidates:
        if (c / "cnxt_v0_fold0.pt").exists():
            base_dir = c
            break
    if base_dir is None:
        for c in glob.glob("**/cnxt_v0_fold0.pt", recursive=True):
            base_dir = Path(c).parent
            break
    if base_dir is not None:
        ckpts = [base_dir / f"cnxt_v0_fold{i}.pt" for i in range(3)]
        return [p for p in ckpts if p.exists()]
    return []

def _find_specialist_weights():
    candidates = [
        Path("/kaggle/input/datasets/aleixlopez/rsna-knee-specialist-weights/specialist_rad_fold0.pth"),
        Path("/kaggle/input/rsna-knee-specialist-weights/specialist_rad_fold0.pth"),
    ]
    for c in candidates:
        if c.exists():
            return c
    for c in glob.glob("**/specialist_rad_fold0.pth", recursive=True):
        return Path(c)
    return None

def _run_golden_fusion_v5():
    if not _sub_path.exists():
        print("[golden-fusion-v5] Warning: Foundation submission.csv does not exist. Skipping.", flush=True)
        return

    pub_sub = pd.read_csv(_sub_path, dtype={"StudyInstanceUID": str})
    print(f"[golden-fusion-v5] Loaded foundation submission: {pub_sub.shape}", flush=True)

    cnxt_ckpts = _find_convnext_weights()
    spec_ckpt = _find_specialist_weights()
    print(f"[golden-fusion-v5] ConvNeXt checkpoints found: {len(cnxt_ckpts)}")
    print(f"[golden-fusion-v5] Specialist checkpoint found: {spec_ckpt}")

    if not cnxt_ckpts and spec_ckpt is None:
        print("[golden-fusion-v5] No auxiliary weights found. Preserving 0.943 foundation anchor.", flush=True)
        return

    # Locate test data
    comp_candidates = [
        Path("/kaggle/input/competitions/rsna-knee-abnormality-detection"),
        Path("/kaggle/input/rsna-knee-abnormality-detection"),
    ]
    comp_root = None
    for c in comp_candidates:
        if (c / "test_series.csv").exists():
            comp_root = c
            break
    if comp_root is None:
        for c in glob.glob("**/test_series.csv", recursive=True):
            comp_root = Path(c).parent
            break

    if comp_root is None or not (comp_root / "test_series.csv").exists():
        print("[golden-fusion-v5] test_series.csv not found. Preserving foundation anchor.", flush=True)
        return

    test_series_df = pd.read_csv(comp_root / "test_series.csv")
    test_series_root = comp_root / "test_series"
    test_uids = list(pub_sub["StudyInstanceUID"].astype(str))
    print(f"[golden-fusion-v5] Total test studies: {len(test_uids)}")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # ========================== 1. CONVNEXT 2.5D READER INFERENCE ==========================
    cnxt_df = None
    if cnxt_ckpts:
        try:
            print("[golden-fusion-v5] Executing ConvNeXt 2.5D Reader Stream...", flush=True)
            # Define KneeNet architecture
            _PLANES = ["Sagittal", "Coronal", "Axial"]
            _MEAN = (0.485, 0.456, 0.406)
            _STD = (0.229, 0.224, 0.225)

            def _make_views(x, res, train, group, fov=0.92):
                n = x.shape[0]
                ns = n // group
                dev = x.device
                x = x.float().div_(255.0)
                ang = torch.zeros(ns, device=dev)
                sc = torch.full((ns,), fov, device=dev)
                tx = ty = torch.zeros(ns, device=dev)
                cos, sin = torch.cos(ang) * sc, torch.sin(ang) * sc
                theta = torch.stack([torch.stack([cos, -sin, tx], 1), torch.stack([sin, cos, ty], 1)], 1)
                theta = theta.repeat_interleave(group, 0)
                grid = F.affine_grid(theta, (n, 3, res, res), align_corners=False)
                if x.shape[-1] > 1.5 * res:
                    x = F.avg_pool2d(x, 2) if x.shape[-1] >= 2 * res else F.interpolate(x, scale_factor=0.75, mode="bilinear", antialias=True)
                x = F.grid_sample(x, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
                mean = torch.tensor(_MEAN, device=dev).view(1, 3, 1, 1)
                std = torch.tensor(_STD, device=dev).view(1, 3, 1, 1)
                return (x - mean) / std

            class KneeNet(nn.Module):
                def __init__(self, backbone="convnext_tiny", pretrained=False, dim=384, depth=2, drop=0.2, drop_path=0.1):
                    super().__init__()
                    kw = {"drop_path_rate": drop_path} if drop_path else {}
                    self.enc = timm.create_model(backbone, pretrained=pretrained, num_classes=0, **kw)
                    self.proj = nn.Sequential(nn.Dropout(drop), nn.Linear(self.enc.num_features, dim), nn.LayerNorm(dim))
                    self.slot_emb = nn.Embedding(6, dim)
                    self.pos_emb = nn.Sequential(nn.Linear(1, dim), nn.GELU(), nn.Linear(dim, dim))
                    layer = nn.TransformerEncoderLayer(dim, 8, dim * 2, dropout=0.1, batch_first=True, norm_first=True)
                    self.tf = nn.TransformerEncoder(layer, depth, enable_nested_tensor=False) if depth else None
                    self.att = nn.Sequential(nn.Linear(dim, 256), nn.Tanh(), nn.Dropout(drop), nn.Linear(256, len(_TARGET_COLS)))
                    self.cls_w = nn.Parameter(torch.randn(len(_TARGET_COLS), dim) * 0.02)
                    self.cls_b = nn.Parameter(torch.zeros(len(_TARGET_COLS)))

                def head(self, feat, mask, pos):
                    b, s, k, _ = feat.shape
                    slot = torch.arange(s, device=feat.device).view(1, s, 1).expand(b, s, k)
                    h = self.proj(feat) + self.slot_emb(slot) + self.pos_emb(pos.unsqueeze(-1).to(feat.dtype))
                    h = h.view(b, s * k, -1)
                    pad = ~mask.unsqueeze(-1).expand(b, s, k).reshape(b, s * k)
                    pad = pad & ~pad.all(1, keepdim=True)
                    if self.tf is not None:
                        h = self.tf(h, src_key_padding_mask=pad)
                    a = self.att(h).float().masked_fill(pad.unsqueeze(-1), -1e4).softmax(1)
                    pooled = torch.einsum("btc,btd->bcd", a.to(h.dtype), h)
                    return (pooled * self.cls_w).sum(-1) + self.cls_b

                def forward(self, x, mask, pos, res=256, train=False):
                    b, s, k = x.shape[:3]
                    keep = mask.view(-1)
                    xv = x.view(b * s, k, *x.shape[3:])[keep].flatten(0, 1)
                    f = self.enc(_make_views(xv, res, train, k).contiguous(memory_format=torch.channels_last))
                    feat = f.new_zeros(b * s, k, f.shape[-1])
                    feat[keep] = f.view(-1, k, f.shape[-1])
                    return self.head(feat.view(b, s, k, -1), mask, pos)

            # Load models
            models = []
            for ckpt in cnxt_ckpts:
                ck = torch.load(ckpt, map_location="cpu", weights_only=False)
                m = KneeNet("convnext_tiny", pretrained=False)
                m.load_state_dict(ck["model"], strict=True)
                models.append(m.to(device).eval())
            print(f"[golden-fusion-v5] Successfully loaded {len(models)} ConvNeXt fold checkpoints!", flush=True)

            # Fast window extractor
            def _cnxt_slot_of(plane, fat_sup):
                return _PLANES.index(plane) + 3 * (1 - int(fat_sup)) if plane in _PLANES else 0

            # Table of candidates per study
            study_table = {}
            for st, sr, fs, pl in zip(test_series_df.StudyInstanceUID, test_series_df.SeriesInstanceUID,
                                      test_series_df.Fat_Suppression, test_series_df.Anatomical_Plane):
                study_table.setdefault(str(st), [[] for _ in range(6)])[_cnxt_slot_of(pl, fs)].append(str(sr))

            def _load_study_cnxt(st_uid, k=12, size=384):
                slots = study_table.get(st_uid, [[] for _ in range(6)])
                x = np.zeros((6, k, 3, size, size), dtype=np.uint8)
                mask = np.zeros(6, dtype=bool)
                pos = np.zeros((6, k), dtype=np.float32)
                for s, cands in enumerate(slots):
                    if not cands: continue
                    s_dir = test_series_root / st_uid / cands[0]
                    if not s_dir.exists(): continue
                    dcm_files = sorted(s_dir.glob("*.dcm"))
                    n = len(dcm_files)
                    if n < 3: continue
                    # Sample k slice centres
                    c_indices = np.linspace(int(0.04 * (n - 1)), int(0.96 * max(n - 1, 1)), k).astype(int)
                    # Load 3-slice slab per centre
                    series_slabs = []
                    for c in c_indices:
                        triplet = [np.clip(c - 1, 0, n - 1), c, np.clip(c + 1, 0, n - 1)]
                        slices = []
                        for idx in triplet:
                            try:
                                d = pydicom.dcmread(str(dcm_files[idx]), stop_before_pixels=False)
                                arr = d.pixel_array.astype(np.float32)
                                p1, p99 = np.percentile(arr, (1, 99.5))
                                arr = np.clip((arr - p1) / max(p99 - p1, 1e-6), 0.0, 1.0)
                                if arr.shape != (size, size):
                                    arr = cv2.resize(arr, (size, size), interpolation=cv2.INTER_LINEAR)
                                slices.append((arr * 255.0).astype(np.uint8))
                            except Exception:
                                slices.append(np.zeros((size, size), dtype=np.uint8))
                        series_slabs.append(np.stack(slices, axis=0))
                    x[s] = np.stack(series_slabs, axis=0)
                    mask[s] = True
                    pos[s] = c_indices / max(n - 1, 1)
                return x, mask, pos

            # Run inference in batches of 4
            all_cnxt_preds = []
            for i in range(0, len(test_uids), 4):
                batch_uids = test_uids[i:i + 4]
                loaded = [_load_study_cnxt(uid) for uid in batch_uids]
                bx = torch.tensor(np.stack([item[0] for item in loaded], 0), device=device)
                bm = torch.tensor(np.stack([item[1] for item in loaded], 0), device=device)
                bp = torch.tensor(np.stack([item[2] for item in loaded], 0), device=device)
                with torch.no_grad():
                    with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                        preds_folds = [m(bx, bm, bp, res=256).sigmoid() for m in models]
                        p_mean = torch.stack(preds_folds, dim=0).mean(0)
                    all_cnxt_preds.append(p_mean.cpu().numpy())
            cnxt_arr = np.concatenate(all_cnxt_preds, axis=0)
            cnxt_df = pd.DataFrame(cnxt_arr, columns=_TARGET_COLS)
            cnxt_df.insert(0, "StudyInstanceUID", test_uids)
            print(f"[golden-fusion-v5] ConvNeXt predictions complete: {cnxt_df.shape}", flush=True)
        except Exception as exc:
            print("[golden-fusion-v5] ConvNeXt inference error:", exc, file=sys.stderr, flush=True)
            traceback.print_exc()
            cnxt_df = None

    # ========================== 2. ASL SPECIALIST INFERENCE ==========================
    spec_df = None
    if spec_ckpt is not None and spec_ckpt.exists():
        try:
            print("[golden-fusion-v5] Executing 384x384 ASL Specialist Stream...", flush=True)
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
                    for i, tgt in enumerate(_TARGET_COLS):
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

            spec_model = SpecialistModel()
            st = torch.load(spec_ckpt, map_location=device)
            st = st.get('state_dict', st.get('model', st))
            st = {k.replace('module.', '').replace('backbone.', 'backbone.'): v for k, v in st.items()}
            spec_model.load_state_dict(st, strict=True)
            spec_model.to(device).eval()

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
                        if p99 > p1: arr = (arr - p1) / (p99 - p1)
                        else: arr = arr - p1
                        arr = np.clip(arr, 0.0, 1.0)
                        if arr.shape != (IMG_SIZE, IMG_SIZE):
                            arr = cv2.resize(arr, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)
                        slices.append(arr)
                    except Exception:
                        slices.append(np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.float32))
                if len(slices) == 3: return np.stack(slices, axis=0)
                return None

            study_groups = {str(uid): g for uid, g in test_series_df.groupby('StudyInstanceUID')}

            def _load_spec_bag(study_uid):
                imgs = np.zeros((6, 3, IMG_SIZE, IMG_SIZE), dtype=np.float32)
                mask = np.zeros(6, dtype=np.float32)
                if study_uid not in study_groups: return imgs, mask
                s_df = study_groups[study_uid]
                for slot_idx, (_, plane, fluid, fs) in enumerate(SLOT_DEFS):
                    cand = s_df[(s_df['Anatomical_Plane'] == plane) & (s_df['Fluid_Sensitive'] == fluid) & (s_df['Fat_Suppression'] == fs)]
                    if len(cand) == 0: cand = s_df[(s_df['Anatomical_Plane'] == plane) & (s_df['Fluid_Sensitive'] == fluid)]
                    if len(cand) == 0: cand = s_df[s_df['Anatomical_Plane'] == plane]
                    if len(cand) > 0:
                        chosen_uid = str(cand.iloc[0]['SeriesInstanceUID'])
                        s_dir = test_series_root / str(study_uid) / chosen_uid
                        if s_dir.exists():
                            slab = _read_spec_slab(s_dir)
                            if slab is not None:
                                imgs[slot_idx] = slab
                                mask[slot_idx] = 1.0
                return imgs, mask

            all_spec_preds = []
            for i in range(0, len(test_uids), 8):
                batch_uids = test_uids[i:i + 8]
                with ThreadPoolExecutor(max_workers=min(len(batch_uids), 8)) as pool:
                    loaded = list(pool.map(_load_spec_bag, batch_uids))
                b_imgs = np.stack([item[0] for item in loaded], axis=0)
                b_mask = np.stack([item[1] for item in loaded], axis=0)
                t_imgs = torch.tensor(b_imgs, dtype=torch.float32, device=device)
                t_mask = torch.tensor(b_mask, dtype=torch.float32, device=device)
                with torch.no_grad():
                    with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                        logits = spec_model(t_imgs, t_mask)
                        probs = torch.sigmoid(logits)
                    all_spec_preds.append(probs.cpu().numpy())

            spec_arr = np.concatenate(all_spec_preds, axis=0)
            spec_df = pd.DataFrame(spec_arr, columns=_TARGET_COLS)
            spec_df.insert(0, "StudyInstanceUID", test_uids)
            print(f"[golden-fusion-v5] Specialist predictions complete: {spec_df.shape}", flush=True)
        except Exception as exc:
            print("[golden-fusion-v5] Specialist inference error:", exc, file=sys.stderr, flush=True)
            traceback.print_exc()
            spec_df = None

    # ========================== 3. THREE-WAY CLINICAL RANK BLEND ==========================
    print("[golden-fusion-v5] Performing 3-Way Rank-Space Clinical Blending...", flush=True)
    final_df = pd.DataFrame({"StudyInstanceUID": test_uids})

    for col in _TARGET_COLS:
        w_f, w_c, w_s = _FUSION_WEIGHTS_3WAY.get(col, _FUSION_WEIGHTS_3WAY["__default__"])
        # Adjust weights dynamically if any stream is absent
        if cnxt_df is None: w_c = 0.0
        if spec_df is None: w_s = 0.0
        tot = w_f + w_c + w_s
        w_f, w_c, w_s = w_f / tot, w_c / tot, w_s / tot

        rf = pub_sub[col].rank(method="average", pct=True).to_numpy(dtype=np.float64) if len(pub_sub) > 1 else pub_sub[col].to_numpy(dtype=np.float64)
        rc = cnxt_df[col].rank(method="average", pct=True).to_numpy(dtype=np.float64) if (cnxt_df is not None and len(cnxt_df) > 1) else (cnxt_df[col].to_numpy(dtype=np.float64) if cnxt_df is not None else rf)
        rs = spec_df[col].rank(method="average", pct=True).to_numpy(dtype=np.float64) if (spec_df is not None and len(spec_df) > 1) else (spec_df[col].to_numpy(dtype=np.float64) if spec_df is not None else rf)

        final_df[col] = w_f * rf + w_c * rc + w_s * rs

    # Final continuous rank normalization across all studies
    if len(final_df) > 1:
        for col in _TARGET_COLS:
            final_df[col] = final_df[col].rank(method="average", pct=True)

    # Verification
    assert list(final_df.columns) == ["StudyInstanceUID"] + _TARGET_COLS, "Column mismatch"
    assert len(final_df) == len(pub_sub), "Row count mismatch"
    assert not final_df.isnull().any().any(), "NaN or Null values detected"
    for col in _TARGET_COLS:
        assert (final_df[col] >= 0.0).all() and (final_df[col] <= 1.0).all(), f"Out of bounds in {col}"

    tmp_path = _sub_path.with_suffix(".csv.tmp")
    with tmp_path.open("w") as fh:
        final_df.to_csv(fh, index=False)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp_path, _sub_path)
    print(f"[golden-fusion-v5] ✅ 6-Pillar Golden Fusion v5 deployed to {_sub_path}!", flush=True)
    print(final_df.head(), flush=True)

try:
    _run_golden_fusion_v5()
except Exception as exc:
    print("[golden-fusion-v5] Global Error during fusion:", exc, file=sys.stderr, flush=True)
    traceback.print_exc()
    print("[golden-fusion-v5] Preserving foundation submission.csv safely.", flush=True)
'''


def update_golden_fusion_notebook():
    nb_path = Path("notebooks/submission/golden_fusion/rsna-knee-golden-fusion.ipynb")
    with open(nb_path) as f:
        nb = json.load(f)

    # Compile check
    compile(CELL_31_V5_CODE, "cell_31_v5", "exec")
    print("CELL_31_V5_CODE compiles cleanly!")

    nb["cells"][31]["source"] = [line + "\n" for line in CELL_31_V5_CODE.splitlines()]

    with open(nb_path, "w") as f:
        json.dump(nb, f, indent=1)

    print(f"Successfully deployed Golden Fusion v5 to {nb_path}!")


if __name__ == "__main__":
    update_golden_fusion_notebook()
