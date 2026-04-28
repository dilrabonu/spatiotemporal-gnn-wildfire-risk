"""
Phase 5A — Run GNN Models (GCN, GraphSAGE, GAT)
"""

import argparse
import torch
import time
import pandas as pd
from pathlib import Path

from torch.optim import Adam
import torch.nn.functional as F

from wildfire_gnn.models.gnn_models import GCNModel, GraphSAGEModel, GATModel
from wildfire_gnn.evaluation.metrics import compute_all_metrics


def train(model, data, optimizer):
    model.train()
    optimizer.zero_grad()

    out = model(data.x, data.edge_index)
    loss = F.mse_loss(out[data.train_mask], data.y[data.train_mask])

    loss.backward()
    optimizer.step()

    return loss.item()


@torch.no_grad()
def evaluate(model, data, mask):
    model.eval()
    pred = model(data.x, data.edge_index)
    return pred[mask], data.y[mask]


def run_model(name, model, data, epochs=50, lr=1e-3):
    optimizer = Adam(model.parameters(), lr=lr)

    best_val = float("inf")
    best_pred = None

    start = time.time()

    for epoch in range(epochs):
        train_loss = train(model, data, optimizer)

        val_pred, val_true = evaluate(model, data, data.val_mask)
        val_loss = F.mse_loss(val_pred, val_true).item()

        if val_loss < best_val:
            best_val = val_loss
            test_pred, test_true = evaluate(model, data, data.test_mask)
            best_pred = (test_pred.cpu().numpy(), test_true.cpu().numpy())

        if epoch % 10 == 0:
            print(f"{name} Epoch {epoch} | train={train_loss:.4f} val={val_loss:.4f}")

    fit_time = time.time() - start

    y_pred, y_true = best_pred
    metrics = compute_all_metrics(y_true, y_pred)

    metrics["model"] = name
    metrics["fit_predict_seconds"] = fit_time

    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph_path", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=50)
    args = parser.parse_args()

    data = torch.load(args.graph_path)

    models = {
        "GCN": GCNModel(data.num_features),
        "GraphSAGE": GraphSAGEModel(data.num_features),
        "GAT": GATModel(data.num_features),
    }

    results = []

    for name, model in models.items():
        print(f"\nRunning {name}")
        model = model.to("cpu")

        res = run_model(name, model, data, epochs=args.epochs)
        results.append(res)

    df = pd.DataFrame(results)
    out_path = Path("reports/tables/phase5a_gnn_metrics.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)

    print("\nSaved:", out_path)


if __name__ == "__main__":
    main()