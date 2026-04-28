"""
Phase 4 — Run Baseline Models

Runs:
1. Naive Mean
2. Ridge Regression
3. Random Forest
4. XGBoost
5. Optional CNN baseline

Outputs:
- reports/tables/phase4_baseline_metrics.csv
- reports/tables/phase4_binned_metrics.csv
- reports/tables/phase4_feature_importances.csv
- reports/predictions/phase4_test_predictions.npz
"""

from __future__ import annotations

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys
import json
import time
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from wildfire_gnn.utils.config import load_yaml_config
from wildfire_gnn.utils.reproducibility import set_seed

# IMPORTANT:
# If your file is baseline.py, change this import to:
# from wildfire_gnn.models.baseline import ...
from wildfire_gnn.models.baselines import (
    NaiveMeanBaseline,
    RidgeBaseline,
    RandomForestBaseline,
    XGBoostBaseline,
)

from wildfire_gnn.evaluation.metrics import compute_all_metrics


def parse_args():
    parser = argparse.ArgumentParser(description="Phase 4 — Baseline Models")

    parser.add_argument("--config", default="configs/gnn_config.yaml")
    parser.add_argument("--run-cnn", action="store_true")
    parser.add_argument("--sample-train", type=int, default=0)
    parser.add_argument("--sample-test", type=int, default=0)

    return parser.parse_args()


def load_graph(graph_path: Path):
    print(f"Loading graph: {graph_path}")
    graph = torch.load(graph_path, map_location="cpu", weights_only=False)

    required_attrs = ["x", "y", "y_raw", "train_mask", "val_mask", "test_mask", "pos"]
    for attr in required_attrs:
        assert hasattr(graph, attr), f"Missing graph attribute: {attr}"

    print(f"Nodes     : {graph.num_nodes:,}")
    print(f"Features  : {graph.num_node_features}")
    print(f"Train     : {int(graph.train_mask.sum()):,}")
    print(f"Val       : {int(graph.val_mask.sum()):,}")
    print(f"Test      : {int(graph.test_mask.sum()):,}")

    return graph


def get_split_arrays(graph, sample_train: int = 0, sample_test: int = 0, seed: int = 42):
    rng = np.random.default_rng(seed)

    X = graph.x.cpu().numpy().astype(np.float32)
    y_t = graph.y.cpu().numpy().astype(np.float32).ravel()
    y_raw = graph.y_raw.cpu().numpy().astype(np.float32).ravel()

    train_idx = np.where(graph.train_mask.cpu().numpy())[0]
    val_idx = np.where(graph.val_mask.cpu().numpy())[0]
    test_idx = np.where(graph.test_mask.cpu().numpy())[0]

    if sample_train and sample_train < len(train_idx):
        train_idx = rng.choice(train_idx, sample_train, replace=False)

    if sample_test and sample_test < len(test_idx):
        test_idx = rng.choice(test_idx, sample_test, replace=False)

    data = {
        "X_train": X[train_idx],
        "y_train_t": y_t[train_idx],
        "y_train_raw": y_raw[train_idx],

        "X_val": X[val_idx],
        "y_val_t": y_t[val_idx],
        "y_val_raw": y_raw[val_idx],

        "X_test": X[test_idx],
        "y_test_t": y_t[test_idx],
        "y_test_raw": y_raw[test_idx],

        "train_idx": train_idx,
        "val_idx": val_idx,
        "test_idx": test_idx,
    }

    return data


def load_feature_names(path: Path, n_features: int):
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            names = json.load(f)

        if len(names) == n_features:
            return names

    return [f"feature_{i}" for i in range(n_features)]


def run_model(model, data, feature_names=None):
    print("\n" + "-" * 70)
    print(f"Running model: {model.name}")
    print("-" * 70)

    t0 = time.time()

    if model.name == "XGBoost":
        model.fit(
            data["X_train"],
            data["y_train_t"],
            data["X_val"],
            data["y_val_t"],
            feature_names=feature_names,
        )
    elif model.name == "Random Forest":
        model.fit(
            data["X_train"],
            data["y_train_t"],
            feature_names=feature_names,
        )
    else:
        model.fit(data["X_train"], data["y_train_t"])

    pred_test_t = model.predict(data["X_test"])

    # Important:
    # Here we evaluate predictions in transformed target space only if your baseline.py
    # already inverse-transforms internally, skip this.
    # Otherwise, for safety, use raw-scale approximation below only after inverse_transform.
    #
    # Best practice: if y was QuantileTransformed, inverse-transform predictions before metrics.
    from wildfire_gnn.evaluation.metrics import inverse_transform_predictions

    transformer_path = PROJECT_ROOT / "data" / "features" / "target_transformer.pkl"
    pred_test_raw = inverse_transform_predictions(pred_test_t, str(transformer_path))

    metrics = compute_all_metrics(
        y_true=data["y_test_raw"],
        y_pred=pred_test_raw,
        model_name=model.name,
        verbose=True,
    )

    metrics["fit_predict_seconds"] = round(time.time() - t0, 2)

    return model, metrics, pred_test_raw


def main():
    args = parse_args()

    print("\n" + "=" * 70)
    print("Phase 4 — Baseline Model Training")
    print("=" * 70)

    config = load_yaml_config(PROJECT_ROOT / args.config)
    seed = config.get("training", {}).get("seed", 42)
    set_seed(seed)

    paths = config["paths"]

    graph_path = PROJECT_ROOT / paths.get(
        "graph_data",
        "data/processed/graph_data_enriched.pt",
    )

    feature_names_path = PROJECT_ROOT / "data" / "features" / "feature_names.json"

    reports_dir = PROJECT_ROOT / "reports"
    tables_dir = reports_dir / "tables"
    pred_dir = reports_dir / "predictions"

    tables_dir.mkdir(parents=True, exist_ok=True)
    pred_dir.mkdir(parents=True, exist_ok=True)

    graph = load_graph(graph_path)

    assert graph.num_node_features == 61, (
        f"Expected 61 features, got {graph.num_node_features}. "
        "Check gnn_config.yaml model.in_channels and graph.node_features."
    )

    data = get_split_arrays(
        graph,
        sample_train=args.sample_train,
        sample_test=args.sample_test,
        seed=seed,
    )

    feature_names = load_feature_names(feature_names_path, graph.num_node_features)

    models = [
        NaiveMeanBaseline(),
        RidgeBaseline(alpha=1.0),
        RandomForestBaseline(
            n_estimators=300,
            max_features="sqrt",
            min_samples_leaf=10,
            n_jobs=-1,
            random_state=seed,
        ),
        XGBoostBaseline(
            n_estimators=1000,
            early_stopping_rounds=50,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=10,
            random_state=seed,
        ),
    ]

    all_metrics = []
    all_binned = []
    predictions = {}

    feature_importance_rows = []

    for model in models:
        fitted_model, metrics, pred_raw = run_model(
            model=model,
            data=data,
            feature_names=feature_names,
        )

        predictions[model.name] = pred_raw

        row = {k: v for k, v in metrics.items() if k != "binned"}
        all_metrics.append(row)

        for b in metrics["binned"]:
            b["model"] = model.name
            all_binned.append(b)

        if hasattr(fitted_model, "top_importances"):
            for rank, (feature, importance) in enumerate(
                fitted_model.top_importances(20), start=1
            ):
                feature_importance_rows.append({
                    "model": fitted_model.name,
                    "rank": rank,
                    "feature": feature,
                    "importance": importance,
                })

    metrics_df = pd.DataFrame(all_metrics)
    binned_df = pd.DataFrame(all_binned)
    importance_df = pd.DataFrame(feature_importance_rows)

    metrics_path = tables_dir / "phase4_baseline_metrics.csv"
    binned_path = tables_dir / "phase4_binned_metrics.csv"
    importance_path = tables_dir / "phase4_feature_importances.csv"
    pred_path = pred_dir / "phase4_test_predictions.npz"

    metrics_df.to_csv(metrics_path, index=False)
    binned_df.to_csv(binned_path, index=False)
    importance_df.to_csv(importance_path, index=False)

    np.savez_compressed(
        pred_path,
        y_test_raw=data["y_test_raw"],
        test_idx=data["test_idx"],
        **{k.replace(" ", "_").replace("(", "").replace(")", ""): v for k, v in predictions.items()},
    )

    print("\n" + "=" * 70)
    print("Phase 4 complete")
    print("=" * 70)
    print(f"Saved metrics      : {metrics_path}")
    print(f"Saved binned       : {binned_path}")
    print(f"Saved importances  : {importance_path}")
    print(f"Saved predictions  : {pred_path}")
    print("\nNext run:")
    print("python scripts/phase4_make_figures.py")


if __name__ == "__main__":
    main()