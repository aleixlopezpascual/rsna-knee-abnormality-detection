from __future__ import annotations

import json
from pathlib import Path

CELL_31_CODE = '''# ============================ GOLDEN FUSION: PROPRIETARY SPECIALIST STREAM ============================
# Blend our trained RadImageNet ResNet50 Specialist with the 0.943 foundation anchor
import gc
import glob
import hashlib
import json
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

print("\\n" + "=" * 80 + "\\n[golden-fusion] Initializing Golden Fusion v4 Specialist Blending Stream\\n" + "=" * 80, flush=True)

def _find_specialist_weights():
    candidates = [
        Path("/kaggle/input/datasets/aleixlopez/rsna-knee-specialist-weights/specialist_rad_fold0.pth"),
        Path("/kaggle/input/rsna-knee-specialist-weights/specialist_rad_fold0.pth"),
    ]
    for c in candidates:
        if c.exists():
            return c
    for c in glob.glob("/kaggle/input/**/specialist_rad_fold0.pth", recursive=True):
        return Path(c)
    return None

_spec_weights = _find_specialist_weights()
print("[golden-fusion] Specialist weights path:", _spec_weights, flush=True)

_FUSION_TARGETS = [
    "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA",
    "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"
]

# Targeted clinical weights: inject specialist signal where it has high ROI
_SPECIALIST_W = {
    "Fracture": 0.25,          # Upweighted by ASL (+0.032 AUC proven)
    "MCL": 0.20,
    "Lateral Meniscus": 0.20,
    "Contusion": 0.15,
    "__default__": 0.05,
}

_sub_path = Path("/kaggle/working/submission.csv")

def _run_specialist_fusion():
    if not _sub_path.exists():
        print("[golden-fusion] Warning: Foundation submission.csv does not exist. Skipping fusion.", flush=True)
        return

    pub_sub = pd.read_csv(_sub_path, dtype={"StudyInstanceUID": str})
    print(f"[golden-fusion] Loaded foundation submission: {pub_sub.shape}", flush=True)

    if _spec_weights is None or not _spec_weights.exists():
        print("[golden-fusion] Specialist weights not found. Preserving 0.943 foundation anchor.", flush=True)
        return

    print("[golden-fusion] Activating Golden Fusion with Specialist weights...", flush=True)

    # 1. Locate competition root and test descriptors
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
        for c in glob.glob("/kaggle/input/**/test_series.csv", recursive=True):
            comp_root = Path(c).parent
            break

    if comp_root is None or not (comp_root / "test_series.csv").exists():
        print("[golden-fusion] test_series.csv not found. Preserving foundation anchor.", flush=True)
        return

    print(f"[golden-fusion] Competition root: {comp_root}", flush=True)
    test_series_df = pd.read_csv(comp_root / "test_series.csv")
    test_series_root = comp_root / "test_series"
    test_uids = list(pub_sub["StudyInstanceUID"].astype(str))
    print(f"[golden-fusion] Total studies to evaluate: {len(test_uids)}")

    # 2. Define Architecture
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
            for i, tgt in enumerate(_FUSION_TARGETS):
                for s_idx in SLOT_PRIOR_TABLE.get(tgt, ()):
                    if s_idx < n_slots:
                        prior_m[i, s_idx] = 0.55
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

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"[golden-fusion] Instantiating SpecialistModel on {device}...")
    model = SpecialistModel()
    st = torch.load(_spec_weights, map_location=device)
    st = st.get('state_dict', st.get('model', st))
    st = {k.replace('module.', '').replace('backbone.', 'backbone.'): v for k, v in st.items()}
    model.load_state_dict(st, strict=True)
    model.to(device).eval()
    print("[golden-fusion] Loaded specialist weights strictly!", flush=True)

    # 3. Fast DICOM Processing
    IMG_SIZE = 384
    SLOT_DEFS = [
        ('SAG_FLUID_FS', 'Sagittal', 1, 1),
        ('COR_FLUID_FS', 'Coronal', 1, 1),
        ('AX_FLUID_FS', 'Axial', 1, 1),
        ('SAG_FLUID_NOFS', 'Sagittal', 1, 0),
        ('COR_T1', 'Coronal', 0, 0),
        ('SAG_T1', 'Sagittal', 0, 0),
    ]

    def _read_series_slab(s_dir):
        dcm_files = sorted(s_dir.glob('*.dcm'))
        n = len(dcm_files)
        if n == 0:
            return None
        indices = np.linspace(int(0.20 * n), int(0.80 * max(n - 1, 1)), 3).astype(int)
        slices = []
        for idx in indices:
            try:
                d = pydicom.dcmread(str(dcm_files[idx]), stop_before_pixels=False)
                arr = d.pixel_array.astype(np.float32)
                p1, p99 = np.percentile(arr, (1, 99))
                if p99 > p1:
                    arr = (arr - p1) / (p99 - p1)
                else:
                    arr = arr - p1
                arr = np.clip(arr, 0.0, 1.0)
                if arr.shape != (IMG_SIZE, IMG_SIZE):
                    arr = cv2.resize(arr, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)
                slices.append(arr)
            except Exception:
                slices.append(np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.float32))
        if len(slices) == 3:
            return np.stack(slices, axis=0)
        return None

    study_groups = {str(uid): g for uid, g in test_series_df.groupby('StudyInstanceUID')}

    def _load_study_bag(study_uid):
        imgs = np.zeros((6, 3, IMG_SIZE, IMG_SIZE), dtype=np.float32)
        mask = np.zeros(6, dtype=np.float32)
        if study_uid not in study_groups:
            return imgs, mask
        s_df = study_groups[study_uid]
        for slot_idx, (_, plane, fluid, fs) in enumerate(SLOT_DEFS):
            cand = s_df[(s_df['Anatomical_Plane'] == plane) &
                        (s_df['Fluid_Sensitive'] == fluid) &
                        (s_df['Fat_Suppression'] == fs)]
            if len(cand) == 0:
                cand = s_df[(s_df['Anatomical_Plane'] == plane) &
                            (s_df['Fluid_Sensitive'] == fluid)]
            if len(cand) == 0:
                cand = s_df[s_df['Anatomical_Plane'] == plane]
            if len(cand) > 0:
                chosen_uid = str(cand.iloc[0]['SeriesInstanceUID'])
                s_dir = test_series_root / str(study_uid) / chosen_uid
                if s_dir.exists():
                    slab = _read_series_slab(s_dir)
                    if slab is not None:
                        imgs[slot_idx] = slab
                        mask[slot_idx] = 1.0
        return imgs, mask

    # 4. Streamed Inference
    BATCH_SIZE = 8
    all_spec_preds = []
    t_infer_start = time.time()
    print(f"[golden-fusion] Running streamed inference (batch={BATCH_SIZE})...", flush=True)

    for i in range(0, len(test_uids), BATCH_SIZE):
        batch_uids = test_uids[i:i + BATCH_SIZE]
        with ThreadPoolExecutor(max_workers=min(len(batch_uids), 8)) as pool:
            loaded = list(pool.map(_load_study_bag, batch_uids))

        b_imgs = np.stack([item[0] for item in loaded], axis=0)
        b_mask = np.stack([item[1] for item in loaded], axis=0)

        t_imgs = torch.tensor(b_imgs, dtype=torch.float32, device=device)
        t_mask = torch.tensor(b_mask, dtype=torch.float32, device=device)

        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                logits = model(t_imgs, t_mask)
                probs = torch.sigmoid(logits)
            all_spec_preds.append(probs.cpu().numpy())

        if (i // BATCH_SIZE) % 10 == 0 or (i + BATCH_SIZE) >= len(test_uids):
            print(f"  [{min(i + BATCH_SIZE, len(test_uids)):4d}/{len(test_uids):4d}] Evaluated in {time.time() - t_infer_start:.1f}s", flush=True)

    spec_preds_arr = np.concatenate(all_spec_preds, axis=0)
    spec_df = pd.DataFrame(spec_preds_arr, columns=_FUSION_TARGETS)
    spec_df.insert(0, 'StudyInstanceUID', test_uids)
    print(f"[golden-fusion] Specialist predictions shape: {spec_df.shape}")

    # 5. Dual-Stream Rank Blending
    print("[golden-fusion] Fusing Foundation + Specialist in Rank Space...", flush=True)
    blended_df = pd.DataFrame({'StudyInstanceUID': test_uids})

    for col in _FUSION_TARGETS:
        w_spec = _SPECIALIST_W.get(col, _SPECIALIST_W["__default__"])
        w_pub = 1.0 - w_spec

        p_pub = pub_sub[col].to_numpy(dtype=np.float64)
        p_spec = spec_df[col].to_numpy(dtype=np.float64)

        if len(pub_sub) > 1:
            rank_pub = pd.Series(p_pub).rank(method='average', pct=True).to_numpy()
            rank_spec = pd.Series(p_spec).rank(method='average', pct=True).to_numpy()
        else:
            rank_pub = p_pub
            rank_spec = p_spec

        blended_val = w_pub * rank_pub + w_spec * rank_spec
        blended_df[col] = blended_val

    # Final continuous rank normalization across all studies
    if len(blended_df) > 1:
        for col in _FUSION_TARGETS:
            blended_df[col] = blended_df[col].rank(method='average', pct=True)

    # 6. Rigorous Verification
    assert list(blended_df.columns) == ['StudyInstanceUID'] + _FUSION_TARGETS, 'Column schema mismatch'
    assert len(blended_df) == len(pub_sub), 'Row count mismatch'
    assert not blended_df.isnull().any().any(), 'NaN or Null values detected'
    for col in _FUSION_TARGETS:
        assert (blended_df[col] >= 0.0).all() and (blended_df[col] <= 1.0).all(), f'Value out of bounds in {col}'

    # 7. Atomic Publication
    tmp_path = _sub_path.with_suffix('.csv.tmp')
    with tmp_path.open('w') as fh:
        blended_df.to_csv(fh, index=False)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp_path, _sub_path)
    print(f"[golden-fusion] Successfully blended Golden Fusion v4 into {_sub_path}!", flush=True)
    print(blended_df.head(), flush=True)

try:
    _run_specialist_fusion()
except Exception as exc:
    print("[golden-fusion] Error during specialist fusion:", exc, file=sys.stderr, flush=True)
    traceback.print_exc()
    print("[golden-fusion] Preserving foundation submission.csv safely.", flush=True)
'''


def update_notebook():
    nb_path = Path("notebooks/submission/golden_fusion/rsna-knee-golden-fusion.ipynb")
    with open(nb_path) as f:
        nb = json.load(f)

    # Compile check before writing
    compile(CELL_31_CODE, "cell_31", "exec")
    print("CELL_31_CODE compiles cleanly!")

    nb["cells"][31]["source"] = [line + "\n" for line in CELL_31_CODE.splitlines()]

    with open(nb_path, "w") as f:
        json.dump(nb, f, indent=1)
    print(f"Successfully updated {nb_path}!")


if __name__ == "__main__":
    update_notebook()
