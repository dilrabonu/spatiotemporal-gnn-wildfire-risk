"""
Phase 5A — Deterministic GNN Architecture

Run quick smoke test:
    python scripts/phase5a_run_gnn.py --model gcn --epochs 5 --device cpu

Run full GCN:
    python scripts/phase5a_run_gnn.py --model gcn --epochs 50 --device cpu

Run full GraphSAGE:
    python scripts/phase5a_run_gnn.py --model graphsage --epochs 50 --device cpu

Run full GAT:
    python scripts/phase5a_run_gnn.py --model gat --epochs 50 --device cpu

Purpose
-------
This script trains deterministic GNNs and evaluates them using the same
metric framework as Phase 4 and Phase 4B.

Important
---------
The model predicts transformed target values.
Predictions are inverse-transformed before reporting R², MAE, Spearman,
Brier, ECE, and binned metrics.
"""

from __future__ import annotations

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys
import time
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from wildfire_gnn.models.gnn_models import build_gnn_model
from wildfire_gnn.evaluation.metrics import compute_all_metrics, inverse_transform_predictions


def parse_args():
    p = argparse.ArgumentParser(description="Phase 5A — Deterministic GNN Training")

    p.add_argument("--graph-path", default="data/processed/graph_data_enriched.pt")
    p.add_argument("--model", default="gcn", choices=["gcn", "graphsage", "gat"])

    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--hidden-channels", type=int, default=64)
    p.add_argument("--num-layers", type=int, default=3)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.3)

    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--patience", type=int, default=12)

    p.add_argument("--device", default="cpu")
    p.add_argument("--seed", type=int, default=42)

    return p.parse_args()


def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_graph(graph_path: Path):
    print(f"Loading graph: {graph_path}")

    graph = torch.load(graph_path, map_location="cpu", weights_only=False)

    required_attrs = ["x", "y", "y_raw", "edge_index", "train_mask", "val_mask", "test_mask"]
    for attr in required_attrs:
        assert hasattr(graph, attr), f"Missing graph attribute: {attr}"

    print(f"Nodes     : {graph.num_nodes:,}")
    print(f"Features  : {graph.num_node_features}")
    print(f"Edges     : {graph.num_edges:,}")
    print(f"Train     : {int(graph.train_mask.sum()):,}")
    print(f"Val       : {int(graph.val_mask.sum()):,}")
    print(f"Test      : {int(graph.test_mask.sum()):,}")

    assert graph.num_node_features == 61, (
        f"Expected 61 node features, got {graph.num_node_features}."
    )

    return graph


def move_graph_to_device(graph, device: torch.device):
    graph.x = graph.x.to(device)
    graph.y = graph.y.to(device)
    graph.y_raw = graph.y_raw.to(device)
    graph.edge_index = graph.edge_index.to(device)
    graph.train_mask = graph.train_mask.to(device)
    graph.val_mask = graph.val_mask.to(device)
    graph.test_mask = graph.test_mask.to(device)
    return graph


def mse_on_mask(pred: torch.Tensor, y: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(pred[mask], y.view(-1)[mask])


@torch.no_grad()
def evaluate_val_loss(model, graph) -> float:
    model.eval()
    pred = model(graph.x, graph.edge_index)
    val_loss = mse_on_mask(pred, graph.y, graph.val_mask)
    return float(val_loss.item())


@torch.no_grad()
def predict_on_mask(model, graph, mask: torch.Tensor) -> np.ndarray:
    model.eval()
    pred = model(graph.x, graph.edge_index)
    return pred[mask].detach().cpu().numpy().ravel()


def train_one_model(args, graph, device: torch.device):
    model = build_gnn_model(
        model_name=args.model,
        in_channels=graph.num_node_features,
        hidden_channels=args.hidden_channels,
        num_layers=args.num_layers,
        heads=args.heads,
        dropout=args.dropout,
    ).to(device)

    print("\nModel architecture:")
    print(model)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=5,
    )

    best_val_loss = float("inf")
    best_state = None
    patience_count = 0
    history = []

    start = time.time()

    print("\nTraining started...")

    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad()

        pred = model(graph.x, graph.edge_index)
        train_loss = mse_on_mask(pred, graph.y, graph.train_mask)

        train_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
        optimizer.step()

        val_loss = evaluate_val_loss(model, graph)
        scheduler.step(val_loss)

        row = {
            "epoch": epoch,
            "train_loss": float(train_loss.item()),
            "val_loss": float(val_loss),
            "lr": optimizer.param_groups[0]["lr"],
        }
        history.append(row)

        if epoch == 1 or epoch % 5 == 0:
            print(
                f"Epoch {epoch:03d}/{args.epochs} | "
                f"train_loss={row['train_loss']:.5f} | "
                f"val_loss={row['val_loss']:.5f} | "
                f"lr={row['lr']:.2e}"
            )

        if val_loss < best_val_loss - 1e-6:
            best_val_loss = val_loss
            patience_count = 0
            best_state = {
                k: v.detach().cpu().clone()
                for k, v in model.state_dict().items()
            }
        else:
            patience_count += 1

        if patience_count >= args.patience:
            print(f"Early stopping at epoch {epoch}")
            break

    fit_time = time.time() - start

    if best_state is not None:
        model.load_state_dict(best_state)

    print(f"\nTraining finished in {fit_time:.1f}s")
    print(f"Best val_loss: {best_val_loss:.5f}")

    return model, history, fit_time, best_val_loss


def main():
    args = parse_args()
    set_seed(args.seed)

    print("\n" + "=" * 75)
    print("Phase 5A — Deterministic GNN Architecture")
    print("=" * 75)

    device = torch.device(args.device)
    graph_path = PROJECT_ROOT / args.graph_path
    transformer_path = PROJECT_ROOT / "data" / "features" / "target_transformer.pkl"

    tables_dir = PROJECT_ROOT / "reports" / "tables"
    pred_dir = PROJECT_ROOT / "reports" / "predictions"
    ckpt_dir = PROJECT_ROOT / "reports" / "checkpoints"

    tables_dir.mkdir(parents=True, exist_ok=True)
    pred_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    graph = load_graph(graph_path)
    graph = move_graph_to_device(graph, device)

    model, history, fit_time, best_val_loss = train_one_model(args, graph, device)

    model_tag = f"phase5a_{args.model}"

    ckpt_path = ckpt_dir / f"{model_tag}_best.pt"
    torch.save(model.state_dict(), ckpt_path)

    print("\nPredicting on test split...")

    pred_test_t = predict_on_mask(model, graph, graph.test_mask)

    y_test_raw = graph.y_raw[graph.test_mask].detach().cpu().numpy().ravel()
    y_test_t = graph.y[graph.test_mask].detach().cpu().numpy().ravel()

    pred_test_raw = inverse_transform_predictions(
        pred_test_t,
        str(transformer_path),
    )

    metrics = compute_all_metrics(
        y_true=y_test_raw,
        y_pred=pred_test_raw,
        model_name=args.model.upper(),
        verbose=True,
    )

    metrics["fit_predict_seconds"] = round(fit_time, 2)
    metrics["best_val_loss"] = best_val_loss
    metrics["epochs_ran"] = len(history)
    metrics["hidden_channels"] = args.hidden_channels
    metrics["num_layers"] = args.num_layers
    metrics["dropout"] = args.dropout
    metrics["lr"] = args.lr
    metrics["weight_decay"] = args.weight_decay
    metrics["device"] = args.device

    metrics_row = {k: v for k, v in metrics.items() if k != "binned"}

    metrics_path = tables_dir / f"{model_tag}_metrics.csv"
    binned_path = tables_dir / f"{model_tag}_binned_metrics.csv"
    history_path = tables_dir / f"{model_tag}_history.csv"
    pred_path = pred_dir / f"{model_tag}_test_predictions.npz"

    pd.DataFrame([metrics_row]).to_csv(metrics_path, index=False)

    binned_df = pd.DataFrame(metrics["binned"])
    binned_df["model"] = args.model.upper()
    binned_df.to_csv(binned_path, index=False)

    pd.DataFrame(history).to_csv(history_path, index=False)

    test_idx = torch.where(graph.test_mask)[0].detach().cpu().numpy()

    np.savez_compressed(
        pred_path,
        y_test_raw=y_test_raw,
        y_test_t=y_test_t,
        pred_test_raw=pred_test_raw,
        pred_test_t=pred_test_t,
        test_idx=test_idx,
    )

    print("\n" + "=" * 75)
    print("Phase 5A complete")
    print("=" * 75)
    print(f"Saved metrics     : {metrics_path}")
    print(f"Saved binned      : {binned_path}")
    print(f"Saved history     : {history_path}")
    print(f"Saved predictions : {pred_path}")
    print(f"Saved checkpoint  : {ckpt_path}")
    print("\nNext run:")
    print("python scripts/phase5a_make_figures.py")


if __name__ == "__main__":
    main()