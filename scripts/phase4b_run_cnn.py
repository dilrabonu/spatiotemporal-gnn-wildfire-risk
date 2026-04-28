"""
Phase 4B — CNN Spatial Baseline

Run quick test:
    python scripts/phase4b_run_cnn.py --sample-train 30000 --sample-test 10000 --epochs 10 --device cpu

Run full:
    python scripts/phase4b_run_cnn.py --epochs 30 --device cpu
"""

from __future__ import annotations

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from wildfire_gnn.models.cnn_baseline import CNNBaseline, extract_patches
from wildfire_gnn.evaluation.metrics import compute_all_metrics, inverse_transform_predictions


def parse_args():
    p = argparse.ArgumentParser(description="Phase 4B — CNN Spatial Baseline")
    p.add_argument("--graph-path", default="data/processed/graph_data_enriched.pt")
    p.add_argument("--aligned-dir", default="data/interim/aligned")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--patch-radius", type=int, default=3)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--patience", type=int, default=7)
    p.add_argument("--device", default="cpu")
    p.add_argument("--sample-train", type=int, default=0)
    p.add_argument("--sample-test", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def sample_indices(idx, sample_size, seed):
    if sample_size and sample_size < len(idx):
        rng = np.random.default_rng(seed)
        return rng.choice(idx, size=sample_size, replace=False)
    return idx


def main():
    args = parse_args()

    print("\n" + "=" * 70)
    print("Phase 4B — CNN Spatial Baseline")
    print("=" * 70)

    graph_path = PROJECT_ROOT / args.graph_path
    aligned_dir = PROJECT_ROOT / args.aligned_dir
    transformer_path = PROJECT_ROOT / "data" / "features" / "target_transformer.pkl"

    tables_dir = PROJECT_ROOT / "reports" / "tables"
    pred_dir = PROJECT_ROOT / "reports" / "predictions"
    checkpoint_dir = PROJECT_ROOT / "reports" / "checkpoints"

    tables_dir.mkdir(parents=True, exist_ok=True)
    pred_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading graph: {graph_path}")
    graph = torch.load(graph_path, map_location="cpu", weights_only=False)

    pos = graph.pos.cpu().numpy().astype(np.int32)
    rows = pos[:, 0]
    cols = pos[:, 1]

    train_idx = np.where(graph.train_mask.cpu().numpy())[0]
    val_idx = np.where(graph.val_mask.cpu().numpy())[0]
    test_idx = np.where(graph.test_mask.cpu().numpy())[0]

    train_idx = sample_indices(train_idx, args.sample_train, args.seed)
    test_idx = sample_indices(test_idx, args.sample_test, args.seed)

    print(f"Train nodes: {len(train_idx):,}")
    print(f"Val nodes  : {len(val_idx):,}")
    print(f"Test nodes : {len(test_idx):,}")

    y_train_t = graph.y[train_idx].cpu().numpy().ravel()
    y_val_t = graph.y[val_idx].cpu().numpy().ravel()
    y_test_raw = graph.y_raw[test_idx].cpu().numpy().ravel()

    print("\n[1/5] Extracting train patches")
    X_train = extract_patches(
        rows_idx=rows[train_idx],
        cols_idx=cols[train_idx],
        aligned_dir=aligned_dir,
        patch_radius=args.patch_radius,
    )

    print("\n[2/5] Extracting validation patches")
    X_val = extract_patches(
        rows_idx=rows[val_idx],
        cols_idx=cols[val_idx],
        aligned_dir=aligned_dir,
        patch_radius=args.patch_radius,
    )

    print("\n[3/5] Extracting test patches")
    X_test = extract_patches(
        rows_idx=rows[test_idx],
        cols_idx=cols[test_idx],
        aligned_dir=aligned_dir,
        patch_radius=args.patch_radius,
    )

    print("\n[4/5] Training CNN")
    cnn = CNNBaseline(
        patch_radius=args.patch_radius,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        patience=args.patience,
        random_state=args.seed,
    )

    cnn.fit(
        X_train_patches=X_train,
        y_train=y_train_t,
        X_val_patches=X_val,
        y_val=y_val_t,
        device_str=args.device,
    )

    print("\n[5/5] Predicting and evaluating")
    pred_t = cnn.predict(X_test)
    pred_raw = inverse_transform_predictions(pred_t, str(transformer_path))

    metrics = compute_all_metrics(
        y_true=y_test_raw,
        y_pred=pred_raw,
        model_name="2D CNN (spatial)",
        verbose=True,
    )

    metrics["fit_predict_seconds"] = round(cnn.fit_time, 2)
    metrics["epochs_requested"] = args.epochs
    metrics["batch_size"] = args.batch_size
    metrics["patch_radius"] = args.patch_radius
    metrics["device"] = args.device
    metrics["n_train"] = len(train_idx)
    metrics["n_val"] = len(val_idx)
    metrics["n_test"] = len(test_idx)

    metrics_row = {k: v for k, v in metrics.items() if k != "binned"}
    metrics_df = pd.DataFrame([metrics_row])

    binned_df = pd.DataFrame(metrics["binned"])
    binned_df["model"] = "2D CNN (spatial)"

    history_df = pd.DataFrame(cnn.history_)

    metrics_path = tables_dir / "phase4b_cnn_metrics.csv"
    binned_path = tables_dir / "phase4b_cnn_binned_metrics.csv"
    history_path = tables_dir / "phase4b_cnn_history.csv"
    pred_path = pred_dir / "phase4b_cnn_test_predictions.npz"
    model_path = checkpoint_dir / "phase4b_cnn_best.pt"

    metrics_df.to_csv(metrics_path, index=False)
    binned_df.to_csv(binned_path, index=False)
    history_df.to_csv(history_path, index=False)

    np.savez_compressed(
        pred_path,
        y_test_raw=y_test_raw,
        pred_cnn_raw=pred_raw,
        pred_cnn_t=pred_t,
        test_idx=test_idx,
    )

    torch.save(cnn.model.state_dict(), model_path)

    print("\n" + "=" * 70)
    print("Phase 4B CNN complete")
    print("=" * 70)
    print(f"Saved metrics     : {metrics_path}")
    print(f"Saved binned      : {binned_path}")
    print(f"Saved history     : {history_path}")
    print(f"Saved predictions : {pred_path}")
    print(f"Saved model       : {model_path}")
    print("\nNext run:")
    print("python scripts/phase4b_make_cnn_figures.py")


if __name__ == "__main__":
    main()