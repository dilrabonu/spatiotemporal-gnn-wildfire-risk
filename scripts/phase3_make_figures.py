"""
Phase 3 — Safe Figure Generation

Generates publication-style visualizations from graph_data_enriched.pt
without crashing Jupyter/VS Code.

Outputs:
    reports/figures/
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import gc
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from torch_geometric.utils import degree


# ============================================================
# Paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

GRAPH_PATH = PROJECT_ROOT / "data" / "processed" / "graph_data_enriched.pt"
FEATURE_NAMES_PATH = PROJECT_ROOT / "data" / "features" / "feature_names.json"
FIG_DIR = PROJECT_ROOT / "reports" / "figures"

FIG_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# Utilities
# ============================================================

def save_fig(fig, filename: str, dpi: int = 150):
    out_path = FIG_DIR / filename
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    gc.collect()
    print(f"✓ Saved: {out_path}")


def load_feature_names(num_features: int):
    if FEATURE_NAMES_PATH.exists():
        with open(FEATURE_NAMES_PATH, "r", encoding="utf-8") as f:
            names = json.load(f)

        if len(names) == num_features:
            return names

        print(
            f"⚠ feature_names length={len(names)} but graph has {num_features} features. "
            "Using fallback names."
        )

    return [f"feature_{i}" for i in range(num_features)]


# ============================================================
# 1. Spatial node distribution
# ============================================================

def plot_spatial_node_distribution(graph):
    print("\n[1/7] Spatial node distribution")

    pos = graph.pos.detach().cpu().numpy()
    rows = pos[:, 0].astype(np.int32)
    cols = pos[:, 1].astype(np.int32)
    y_raw = graph.y_raw.detach().cpu().numpy().ravel()

    n_plot = min(30_000, graph.num_nodes)
    idx = np.random.default_rng(42).choice(graph.num_nodes, size=n_plot, replace=False)

    fig, ax = plt.subplots(figsize=(8, 8))
    sc = ax.scatter(
        cols[idx],
        rows[idx],
        c=y_raw[idx],
        s=0.6,
        cmap="YlOrRd",
        alpha=0.75,
    )

    ax.invert_yaxis()
    ax.set_title(f"Spatial Distribution of Graph Nodes ({n_plot:,} sampled)")
    ax.set_xlabel("Column")
    ax.set_ylabel("Row")
    fig.colorbar(sc, ax=ax, label="Burn Probability")

    save_fig(fig, "p3_node_distribution.png")


# ============================================================
# 2. Target distributions
# ============================================================

def plot_target_distributions(graph):
    print("\n[2/7] Target distributions")

    y_raw = graph.y_raw.detach().cpu().numpy().ravel()
    y_t = graph.y.detach().cpu().numpy().ravel()

    fig, axes = plt.subplots(1, 2, figsize=(13, 4))

    axes[0].hist(y_raw, bins=80, alpha=0.85)
    axes[0].axvline(float(np.mean(y_raw)), linestyle="--", linewidth=2)
    axes[0].set_xlabel("Burn Probability")
    axes[0].set_ylabel("Node count")
    axes[0].set_title(f"Raw Burn Probability\nmean={np.mean(y_raw):.4f}")

    axes[1].hist(y_t, bins=80, alpha=0.85)
    axes[1].axvline(float(np.mean(y_t)), linestyle="--", linewidth=2)
    axes[1].set_xlabel("Transformed Target")
    axes[1].set_ylabel("Node count")
    axes[1].set_title(f"Quantile-Transformed Target\nmean={np.mean(y_t):.4f}, std={np.std(y_t):.4f}")

    fig.tight_layout()
    save_fig(fig, "p3_target_distributions.png")

    assert abs(float(np.mean(y_t))) < 0.5, "Target mean is too far from 0."
    assert 0.5 < float(np.std(y_t)) < 2.0, "Target std is not near 1."
    print("✓ Target transform validated")


# ============================================================
# 3. Feature correlations
# ============================================================

def plot_feature_correlations(graph, feature_names):
    print("\n[3/7] Feature correlations")

    X_train = graph.x[graph.train_mask].detach().cpu().numpy()
    y_train = graph.y_raw[graph.train_mask].detach().cpu().numpy().ravel()

    correlations = []

    for i, fname in enumerate(feature_names):
        col = X_train[:, i]

        if np.std(col) < 1e-8:
            r = 0.0
        else:
            r = float(np.corrcoef(col, y_train)[0, 1])
            if np.isnan(r):
                r = 0.0

        correlations.append(r)

    corr_df = pd.DataFrame({
        "feature": feature_names,
        "pearson_r": correlations,
        "abs_pearson_r": np.abs(correlations),
    }).sort_values("abs_pearson_r", ascending=False)

    csv_path = FIG_DIR / "p3_feature_correlations.csv"
    corr_df.to_csv(csv_path, index=False)
    print(f"✓ Saved: {csv_path}")

    top30 = corr_df.head(30)

    fig, ax = plt.subplots(figsize=(9, 10))
    ax.barh(range(len(top30)), top30["pearson_r"].values)
    ax.set_yticks(range(len(top30)))
    ax.set_yticklabels(top30["feature"].values, fontsize=8)
    ax.axvline(0, linewidth=0.8)
    ax.set_xlabel("Pearson r with Burn Probability")
    ax.set_title("Top 30 Feature Correlations\nTrain split only — no leakage")
    ax.invert_yaxis()

    fig.tight_layout()
    save_fig(fig, "p3_feature_correlations.png")


# ============================================================
# 4. Degree distribution
# ============================================================

def plot_degree_distribution(graph):
    print("\n[4/7] Degree distribution")

    deg = degree(graph.edge_index[0], num_nodes=graph.num_nodes).detach().cpu().numpy()

    print("Degree statistics:")
    print(f"  min  : {int(deg.min())}")
    print(f"  max  : {int(deg.max())}")
    print(f"  mean : {deg.mean():.2f}")
    print(f"  std  : {deg.std():.2f}")

    unique, counts = np.unique(deg.astype(int), return_counts=True)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(unique, counts, width=0.7)
    ax.set_xlabel("Node degree")
    ax.set_ylabel("Number of nodes")
    ax.set_xticks(list(range(0, 9)))
    ax.set_title("8-Connected Pixel Grid Degree Distribution")

    fig.tight_layout()
    save_fig(fig, "p3_degree_distribution.png")


# ============================================================
# 5. Split distribution
# ============================================================

def plot_split_distribution(graph):
    print("\n[5/7] Split distribution")

    splits = {
        "Train": int(graph.train_mask.sum()),
        "Validation": int(graph.val_mask.sum()),
        "Test": int(graph.test_mask.sum()),
    }

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(splits.keys(), splits.values())
    ax.set_title("Graph Node Split Distribution")
    ax.set_xlabel("Split")
    ax.set_ylabel("Number of nodes")

    for i, v in enumerate(splits.values()):
        ax.text(i, v, f"{v:,}", ha="center", va="bottom", fontsize=9)

    fig.tight_layout()
    save_fig(fig, "p3_split_distribution.png")


# ============================================================
# 6. Split spatial map
# ============================================================

def plot_split_spatial_map(graph):
    print("\n[6/7] Spatial split map")

    pos = graph.pos.detach().cpu().numpy()
    rows = pos[:, 0].astype(np.int32)
    cols = pos[:, 1].astype(np.int32)

    split_id = np.zeros(graph.num_nodes, dtype=np.int8)
    split_id[graph.train_mask.detach().cpu().numpy()] = 0
    split_id[graph.val_mask.detach().cpu().numpy()] = 1
    split_id[graph.test_mask.detach().cpu().numpy()] = 2

    n_plot = min(50_000, graph.num_nodes)
    idx = np.random.default_rng(123).choice(graph.num_nodes, size=n_plot, replace=False)

    fig, ax = plt.subplots(figsize=(8, 8))
    sc = ax.scatter(
        cols[idx],
        rows[idx],
        c=split_id[idx],
        s=0.6,
        alpha=0.75,
    )

    ax.invert_yaxis()
    ax.set_title(f"Geographic Train/Val/Test Split ({n_plot:,} sampled)")
    ax.set_xlabel("Column")
    ax.set_ylabel("Row")

    cbar = fig.colorbar(sc, ax=ax, ticks=[0, 1, 2])
    cbar.ax.set_yticklabels(["Train", "Validation", "Test"])

    save_fig(fig, "p3_spatial_split_map.png")


# ============================================================
# 7. Selected feature histograms
# ============================================================

def plot_selected_feature_histograms(graph, feature_names):
    print("\n[7/7] Selected feature histograms")

    X = graph.x.detach().cpu().numpy()

    preferred_keywords = [
        "CFL",
        "FSP",
        "Ignition",
        "Struct",
        "dem_elevation",
        "dem_slope",
        "dem_twi",
    ]

    selected = []

    for keyword in preferred_keywords:
        for i, name in enumerate(feature_names):
            if keyword.lower() in name.lower() and i not in selected:
                selected.append(i)
                break

    if len(selected) < 6:
        for i in range(min(graph.num_node_features, 6)):
            if i not in selected:
                selected.append(i)

    selected = selected[:8]

    for idx in selected:
        name = feature_names[idx]
        values = X[:, idx]

        fig, ax = plt.subplots(figsize=(7, 4))
        ax.hist(values, bins=80)
        ax.set_title(f"Feature Distribution: {name}")
        ax.set_xlabel("Value")
        ax.set_ylabel("Node count")

        safe_name = (
            name.replace("/", "_")
                .replace("\\", "_")
                .replace(" ", "_")
                .replace("×", "x")
        )

        fig.tight_layout()
        save_fig(fig, f"p3_feature_{idx}_{safe_name}.png")


# ============================================================
# Main
# ============================================================

def main():
    print("=" * 60)
    print("Phase 3 — Safe Figure Generation")
    print("=" * 60)

    if not GRAPH_PATH.exists():
        raise FileNotFoundError(f"Graph not found: {GRAPH_PATH}")

    graph = torch.load(GRAPH_PATH, map_location="cpu", weights_only=False)
    feature_names = load_feature_names(graph.num_node_features)

    print(f"Loaded graph : {GRAPH_PATH}")
    print(f"Nodes        : {graph.num_nodes:,}")
    print(f"Features     : {graph.num_node_features}")
    print(f"Edges        : {graph.num_edges:,}")
    print(f"Figures dir  : {FIG_DIR}")

    plot_spatial_node_distribution(graph)
    plot_target_distributions(graph)
    plot_feature_correlations(graph, feature_names)
    plot_degree_distribution(graph)
    plot_split_distribution(graph)
    plot_split_spatial_map(graph)
    plot_selected_feature_histograms(graph, feature_names)

    print("\n" + "=" * 60)
    print("ALL PHASE 3 FIGURES GENERATED SUCCESSFULLY")
    print(f"Saved to: {FIG_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()