from __future__ import annotations

import argparse
import json
from pathlib import Path

CODE_BODY = '''import os, sys, time, glob, random, hashlib, gc
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import numpy as np, pandas as pd
import pydicom, cv2, torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import timm

def seed_everything(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

seed_everything(42)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

TARGET_COLS = [
    "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus",
    "Medial OA", "Lateral OA", "PF OA", "Effusion",
    "Synovitis", "Baker's", "Contusion", "Fracture"
]

IMG_SIZE = 224
N_SLOTS = 6
SLOT_DEFS = [
    ("SAG_FLUID_FS", "Sagittal", 1, 1),
    ("COR_FLUID_FS", "Coronal", 1, 1),
    ("AX_FLUID_FS", "Axial", 1, 1),
    ("SAG_FLUID_NOFS", "Sagittal", 1, 0),
    ("COR_T1", "Coronal", 0, 0),
    ("SAG_T1", "Sagittal", 0, 0),
]

# 1. Asymmetric Focal Loss with Soft/Continuous Target Support
class AsymmetricFocalLoss(nn.Module):
    def __init__(self, gamma_neg=4.0, gamma_pos=1.0, clip=0.05, eps=1e-8, class_weights=None):
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.eps = eps
        if class_weights is not None:
            self.register_buffer("class_weights", class_weights.float())
        else:
            self.class_weights = None

    def forward(self, logits, targets, sample_weights=None):
        p = torch.sigmoid(logits)
        xs_pos = p
        xs_neg = 1.0 - p
        if self.clip is not None and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1.0)

        los_pos = targets * torch.log(xs_pos.clamp(min=self.eps))
        los_neg = (1.0 - targets) * torch.log(xs_neg.clamp(min=self.eps))
        loss = los_pos + los_neg

        if self.gamma_neg > 0 or self.gamma_pos > 0:
            pt = p * targets + (1.0 - p) * (1.0 - targets)
            gamma = self.gamma_pos * targets + self.gamma_neg * (1.0 - targets)
            loss = loss * ((1.0 - pt.clamp(min=self.eps)) ** gamma)

        loss = -loss
        if self.class_weights is not None:
            loss = loss * self.class_weights.unsqueeze(0)
        if sample_weights is not None:
            loss = loss * sample_weights

        return loss.mean()

# 2. Model Architecture: DINOv2 ViT-Small with Dynamic Spatial Sizing + SlotHead
SLOT_PRIOR_TABLE = {
    'ACL': (0, 3, 5), 'MCL': (1, 4), 'Medial Meniscus': (0, 1, 3, 4),
    'Lateral Meniscus': (0, 1, 3, 4), 'Medial OA': (1, 4, 5),
    'Lateral OA': (1, 4, 5), 'PF OA': (0, 2, 5), 'Effusion': (0, 2),
    'Synovitis': (0, 2), "Baker's": (0,), 'Contusion': (0, 1, 2),
    'Fracture': (0, 1, 2, 4, 5),
}

class SlotHead(nn.Module):
    def __init__(self, dim=384, n_slots=6, n_targets=12, hidden=256, dropout=0.2):
        super().__init__()
        self.hidden = hidden
        self.proj = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, hidden), nn.GELU())
        self.slot_emb = nn.Parameter(torch.randn(n_slots, hidden) * 0.02)
        self.query = nn.Parameter(torch.randn(n_targets, hidden) * 0.02)
        self.drop = nn.Dropout(dropout)
        self.out = nn.Linear(hidden, n_targets)
        prior_m = torch.zeros(n_targets, n_slots)
        for i, tgt in enumerate(TARGET_COLS):
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

class DinoKneeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = timm.create_model('vit_small_patch14_dinov2', pretrained=False, num_classes=0, img_size=IMG_SIZE, dynamic_img_size=True)
        self.head = SlotHead(dim=self.backbone.num_features, n_slots=6, n_targets=12)
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 1, 3, 1, 1))

    def forward(self, imgs, mask):
        b, s, c, h, w = imgs.shape
        x = (imgs - self.mean) / self.std
        feat = self.backbone(x.view(b * s, c, h, w)).view(b, s, -1)
        return self.head(feat, mask)

print("Architecture and loss definitions initialized successfully!")
'''


def build_dinov2_training_notebook(fold: int = 0, epochs: int = 8, out_path: Path | None = None) -> Path:
    out_dir = Path("notebooks/train_dinov2")
    out_dir.mkdir(parents=True, exist_ok=True)
    if out_path is None:
        out_path = out_dir / f"rsna-knee-dinov2-fold{fold}.ipynb"

    cells = [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                f"# 🦖 RSNA Knee Abnormality Detection: DINOv2 Vision Transformer (Fold {fold})\n",
                "\n",
                f"Executes training of the **3rd Orthogonal Vision Family (DINOv2 ViT-Small)** on **Fold {fold}**:\n",
                "- **Goal:** Break the $\\rho \\approx 0.88$ noise correlation barrier of CNN/CoAtNet models.\n",
                "- **Labels:** Trained on continuous calibrated clinical probabilities $[0.0, 1.0]$.\n",
                "- **Loss:** Asymmetric Focal Loss (ASL) with minority margin shifting.\n",
                "- **Architecture:** DINOv2 ViT-S + 6-slot anatomical `SlotHead` attention.\n",
            ],
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [line + "\n" for line in CODE_BODY.splitlines()],
        },
    ]

    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
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

    with open(out_path, "w") as f:
        json.dump(nb, f, indent=1)

    print(f"Generated {out_path}!")

    meta = {
        "id": f"aleixlopez/rsna-knee-dinov2-fold{fold}",
        "title": f"RSNA Knee DINOv2 Fold {fold}",
        "code_file": out_path.name,
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "enable_tpu": False,
        "enable_internet": False,
        "keywords": ["gpu"],
        "dataset_sources": [
            "aleixlopez/rsna-knee-specialist-weights",
        ],
        "kernel_sources": [],
        "competition_sources": ["rsna-knee-abnormality-detection"],
        "model_sources": [],
    }
    with open(out_dir / "kernel-metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    return out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=8)
    args = parser.parse_args()
    build_dinov2_training_notebook(fold=args.fold, epochs=args.epochs)
