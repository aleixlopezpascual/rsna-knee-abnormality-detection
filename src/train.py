from __future__ import annotations

import argparse
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from src.config import INPUT_DIR, MODELS_DIR, OOF_DIR, SEED, TARGET_COLS
from src.dataset import KneeMultiSlotDataset, generate_report_grouped_folds
from src.metrics import compute_competition_metric, compute_detailed_metrics
from src.models import KneeMultiSlotModel
from src.report_reader import populate_labels_from_reports


def seed_everything(seed: int = SEED) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device(requested: str = "auto") -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def compute_weighted_bce_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    """Compute binary cross entropy loss weighted by report confidence weights."""
    bce = nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    weighted_bce = bce * weights
    return weighted_bce.mean()


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    use_amp: bool = False,
) -> float:
    model.train()
    total_loss = 0.0
    scaler = torch.amp.GradScaler("cuda", enabled=(use_amp and device.type == "cuda"))

    for batch in loader:
        imgs = batch["imgs"].to(device)
        mask = batch["mask"].to(device)
        targets = batch["targets"].to(device)
        weights = batch["weights"].to(device)

        optimizer.zero_grad()
        with torch.autocast(device_type=device.type, enabled=use_amp):
            logits = model(imgs, mask)
            loss = compute_weighted_bce_loss(logits, targets, weights)

        if use_amp and device.type == "cuda":
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        total_loss += float(loss.item())

    return total_loss / max(1, len(loader))


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[float, dict[str, float], np.ndarray, np.ndarray]:
    model.eval()
    all_preds = []
    all_targets = []

    for batch in loader:
        imgs = batch["imgs"].to(device)
        mask = batch["mask"].to(device)
        targets = batch["targets"]

        logits = model(imgs, mask)
        preds = torch.sigmoid(logits).cpu().numpy()

        all_preds.append(preds)
        all_targets.append(targets.numpy())

    y_pred = np.vstack(all_preds)
    y_true = np.vstack(all_targets)

    macro_auc = compute_competition_metric(y_true, y_pred, target_cols=TARGET_COLS)
    detailed = compute_detailed_metrics(y_true, y_pred, target_cols=TARGET_COLS)

    return macro_auc, detailed, y_pred, y_true


def train_fold(
    fold: int,
    df: pd.DataFrame,
    slots_data: dict[str, dict[str, np.ndarray]] | None,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[float, pd.DataFrame]:
    print(f"\n--- Starting Training: Fold {fold} ---")

    train_df = df[df["fold"] != fold].reset_index(drop=True)
    val_df = df[df["fold"] == fold].reset_index(drop=True)

    train_ds = KneeMultiSlotDataset(train_df, slots_data=slots_data, img_size=args.img_size, is_train=True)
    val_ds = KneeMultiSlotDataset(val_df, slots_data=slots_data, img_size=args.img_size, is_train=False)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False
    )

    model = KneeMultiSlotModel(
        backbone_name=args.model,
        pretrained=args.pretrained,
        n_slots=6,
        n_targets=len(TARGET_COLS),
        head_hidden=args.head_hidden,
        dropout=args.dropout,
        prior=True,
    ).to(device)

    # Differential learning rates: slower for backbone, faster for head
    param_groups = [
        {"params": model.backbone.parameters(), "lr": args.lr_backbone},
        {"params": model.head.parameters(), "lr": args.lr},
    ]
    optimizer = torch.optim.AdamW(param_groups, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    best_val_auc = 0.0
    best_preds = None

    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(OOF_DIR, exist_ok=True)
    save_path = Path(MODELS_DIR) / f"{args.model}_fold{fold}_best.pth"

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device, use_amp=args.amp)
        val_auc, detailed, preds, targets = evaluate(model, val_loader, device)
        scheduler.step()

        print(
            f"Epoch {epoch:02d}/{args.epochs:02d} | Train Loss: {train_loss:.4f} | Val Macro-AUC: {val_auc:.5f}"
        )

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_preds = preds
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "val_auc": val_auc,
                    "detailed": detailed,
                    "args": vars(args),
                },
                save_path,
            )

    print(f"Fold {fold} finished. Best Val Macro-AUC: {best_val_auc:.5f}")

    # Build OOF dataframe for this fold
    oof_fold_df = val_df[["StudyInstanceUID"]].copy()
    if best_preds is not None:
        for i, col in enumerate(TARGET_COLS):
            oof_fold_df[f"{col}_pred"] = best_preds[:, i]
            oof_fold_df[f"{col}_true"] = targets[:, i]

    return best_val_auc, oof_fold_df


def main() -> None:
    parser = argparse.ArgumentParser(description="Train RSNA Knee Abnormality Detection Model")
    parser.add_argument("--model", type=str, default="resnet34", help="Backbone architecture")
    parser.add_argument("--pretrained", action="store_true", help="Use ImageNet pretrained weights")
    parser.add_argument("--img-size", type=int, default=224, help="Input resolution")
    parser.add_argument("--fold", type=int, default=0, help="Fold index to train (0-4)")
    parser.add_argument("--n-splits", type=int, default=5, help="Number of cross-validation splits")
    parser.add_argument("--epochs", type=int, default=5, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate for head")
    parser.add_argument("--lr-backbone", type=float, default=1e-5, help="Learning rate for backbone")
    parser.add_argument("--head-hidden", type=int, default=256, help="SlotHead hidden dimension")
    parser.add_argument("--dropout", type=float, default=0.2, help="Head dropout")
    parser.add_argument("--weight-decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--seed", type=int, default=SEED, help="Random seed")
    parser.add_argument("--device", type=str, default="auto", help="Execution device (cuda/mps/cpu/auto)")
    parser.add_argument("--amp", action="store_true", help="Use automatic mixed precision")
    parser.add_argument("--dry-run", action="store_true", help="Run 1 epoch with small synthetic subset")

    args = parser.parse_args()
    seed_everything(args.seed)
    device = get_device(args.device)
    print(f"Using device: {device} | Model: {args.model} | Resolution: {args.img_size}x{args.img_size}")

    train_csv_path = Path(INPUT_DIR) / "train.csv"
    if not train_csv_path.exists() or args.dry_run:
        print("Using dry-run / synthetic test studies...")
        mock_studies = [f"mock_study_{i}" for i in range(20)]
        mock_reports = [f"Report findings {i % 5}" for i in range(20)]
        df = pd.DataFrame({"StudyInstanceUID": mock_studies, "Report": mock_reports})
        for c in TARGET_COLS:
            df[c] = np.random.uniform(0.1, 0.9, size=20)
    else:
        print(f"Loading {train_csv_path}...")
        df = pd.read_csv(train_csv_path)
        print("Populating labels from reports with NLP reader...")
        df = populate_labels_from_reports(df, target_cols=TARGET_COLS)

    # Generate folds
    df = generate_report_grouped_folds(df, target_cols=TARGET_COLS, n_splits=args.n_splits, seed=args.seed)

    # Train fold
    _best_auc, oof_df = train_fold(
        fold=args.fold,
        df=df,
        slots_data=None,  # Real DICOM caches or on-the-fly tensors
        args=args,
        device=device,
    )

    oof_path = Path(OOF_DIR) / f"oof_{args.model}_fold{args.fold}.parquet"
    oof_df.to_parquet(oof_path, index=False)
    print(f"Saved fold OOF predictions to {oof_path}")


if __name__ == "__main__":
    main()
