from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.config import INPUT_DIR, SUBMISSION_DIR, TARGET_COLS
from src.dataset import KneeMultiSlotDataset
from src.models import KneeMultiSlotModel


def validate_submission_df(df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    """Rigorous pre-submission validation asserting all competition requirements."""
    expected_cols = ["StudyInstanceUID"] + list(TARGET_COLS)
    assert list(df.columns) == expected_cols, (
        f"Columns mismatch! Expected: {expected_cols}, got: {list(df.columns)}"
    )
    assert len(df) == len(test_df), (
        f"Row count mismatch! Expected {len(test_df)} rows, got {len(df)}"
    )
    assert not df.isnull().any().any(), "Submission contains null/NaN values!"
    for col in TARGET_COLS:
        assert (df[col] >= 0.0).all() and (df[col] <= 1.0).all(), (
            f"Values in column {col} out of range [0, 1]!"
        )


def apply_anatomical_mirror_tta(
    imgs: torch.Tensor,
    planes: list[str] | tuple[str, ...] | None = None,
) -> torch.Tensor:
    """Apply left/right anatomical mirror transform for test-time augmentation (TTA).

    Proven in nartaa (0.949) to boost single-model performance without parameter increases:
    - Sagittal views: Reverses the slice channel order (dim 2).
    - Coronal/Axial views: Flips image width horizontally (dim -1).

    Args:
        imgs: Input tensor of shape (Batch, Slots, Channels, H, W) or (Batch, Channels, H, W).
        planes: Optional sequence of anatomical planes for each slot.

    Returns:
        Mirrored tensor with identical shape and dtype.
    """
    if imgs.ndim == 5:
        mirrored = imgs.clone()
        if planes is not None:
            for s_idx, plane in enumerate(planes):
                if plane.lower() == "sagittal":
                    mirrored[:, s_idx] = torch.flip(mirrored[:, s_idx], dims=[1])
                else:
                    mirrored[:, s_idx] = torch.flip(mirrored[:, s_idx], dims=[-1])
        else:
            mirrored = torch.flip(mirrored, dims=[-1])
        return mirrored
    elif imgs.ndim == 4:
        return torch.flip(imgs, dims=[-1])
    return imgs


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate RSNA Knee Submissions")
    parser.add_argument("--model", type=str, default="resnet34", help="Backbone architecture")
    parser.add_argument("--weights-path", type=str, default=None, help="Path to .pth checkpoint file")
    parser.add_argument("--test-csv", type=str, default=str(INPUT_DIR / "test.csv"))
    parser.add_argument("--img-size", type=int, default=224, help="Inference resolution")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--output", type=str, default=str(SUBMISSION_DIR / "submission.csv"))
    parser.add_argument("--rank-pct", action="store_true", default=True, help="Apply percentile ranking")
    parser.add_argument("--enable-tta", action="store_true", default=False, help="Enable anatomical mirror TTA")

    args = parser.parse_args()

    test_path = Path(args.test_csv)
    if not test_path.exists():
        print(f"Warning: {test_path} does not exist. Creating stub test DataFrame for validation.")
        test_df = pd.DataFrame({"StudyInstanceUID": ["test_study_01", "test_study_02", "test_study_03"]})
    else:
        test_df = pd.read_csv(test_path)

    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    print(f"Inference device: {device} | Total studies: {len(test_df)}")

    model = KneeMultiSlotModel(
        backbone_name=args.model,
        pretrained=False,
        n_slots=6,
        n_targets=len(TARGET_COLS),
        prior=True,
    ).to(device)

    if args.weights_path and Path(args.weights_path).exists():
        print(f"Loading checkpoint weights from {args.weights_path}...")
        ckpt = torch.load(args.weights_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        print("Running in dry-run / benchmark mode (no weights loaded).")

    model.eval()

    test_ds = KneeMultiSlotDataset(test_df, slots_data=None, img_size=args.img_size, is_train=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    all_preds = []
    with torch.no_grad():
        for batch in test_loader:
            imgs = batch["imgs"].to(device)
            mask = batch["mask"].to(device)
            logits = model(imgs, mask)
            preds = torch.sigmoid(logits)

            if args.enable_tta:
                mirrored_imgs = apply_anatomical_mirror_tta(imgs)
                mirrored_logits = model(mirrored_imgs, mask)
                mirrored_preds = torch.sigmoid(mirrored_logits)
                preds = 0.5 * (preds + mirrored_preds)

            all_preds.append(preds.cpu().numpy())

    preds_arr = np.vstack(all_preds) if all_preds else np.full((len(test_df), len(TARGET_COLS)), 0.5)

    sub_df = pd.DataFrame({"StudyInstanceUID": test_df["StudyInstanceUID"]})
    for i, col in enumerate(TARGET_COLS):
        sub_df[col] = preds_arr[:, i]
        if args.rank_pct and len(sub_df) > 1:
            sub_df[col] = sub_df[col].rank(pct=True)

    validate_submission_df(sub_df, test_df)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sub_df.to_csv(out_path, index=False)
    print(f"✅ Successfully wrote verified submission to {out_path}")
    print(sub_df.head())


if __name__ == "__main__":
    main()
