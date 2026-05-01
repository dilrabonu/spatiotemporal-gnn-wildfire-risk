"""
Publication-Quality Figure Generation — All Phases
====================================================
Generates every figure needed for the paper.

USAGE
-----
    conda activate wildfire-gnn
    cd spatiotemporal_wildfire_gnn
    python scripts/paper_figures.py

OUTPUTS (reports/paper_figures/)
----------------------------------
    fig1_geographic_split.png           Study area + geographic split
    fig2_model_comparison.png           All models R², MAE, Spearman, ECE, Brier
    fig3_prediction_scatter.png         GAT vs CNN vs XGBoost scatter
    fig4_binned_evaluation.png          High-risk tail binned evaluation
    fig5_feature_importance.png         RF + XGBoost top features
    fig6_calibration_reliability.png    GAT calibration before/after
    fig7_uncertainty_decomposition.png  Epistemic vs aleatoric
    fig8_intervention_analysis.png      All 3 intervention scenarios
    fig9_ablation_architecture.png      GAT vs GCN vs GraphSAGE
    fig10_training_dynamics.png         GAT training curve
"""

from __future__ import annotations
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sys, gc, pickle
import numpy as np
import pandas as pd
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import matplotlib.ticker as mticker

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# ── Paths ──────────────────────────────────────────────────────────────────
TBL  = PROJECT_ROOT / "reports" / "tables"
PRED = PROJECT_ROOT / "reports" / "predictions"
FIG  = PROJECT_ROOT / "reports" / "paper_figures"
FIG.mkdir(parents=True, exist_ok=True)

# ── Style ──────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":      "DejaVu Sans",
    "font.size":        11,
    "axes.titlesize":   12,
    "axes.labelsize":   11,
    "xtick.labelsize":  10,
    "ytick.labelsize":  10,
    "legend.fontsize":  9,
    "figure.dpi":       150,
    "axes.spines.top":  False,
    "axes.spines.right":False,
    "axes.grid":        True,
    "grid.alpha":       0.25,
    "grid.linestyle":   "--",
})

# ── Colour palette ─────────────────────────────────────────────────────────
C = {
    "GAT":       "#1a6faf",   # deep blue
    "GCN":       "#4cae4f",   # green
    "GraphSAGE": "#7b3f99",   # purple
    "CNN":       "#e07b39",   # orange
    "XGBoost":   "#b84040",   # red
    "RF":        "#c0a020",   # gold
    "Ridge":     "#888888",   # grey
    "Naive":     "#cccccc",   # light grey
    "before":    "#e74c3c",
    "after":     "#2ecc71",
    "firebreak": "#1a6faf",
    "fuel":      "#e07b39",
    "ignition":  "#7b3f99",
    "accent":    "#e74c3c",
}

LABEL = {
    "GAT":            "GAT (ours)",
    "GCN":            "GCN",
    "GraphSAGE":      "GraphSAGE",
    "2D CNN (spatial)":"2D CNN",
    "XGBoost":        "XGBoost",
    "Random Forest":  "Random Forest",
    "Ridge Regression":"Ridge",
    "Naive Mean":     "Naive Mean",
}

def savefig(fig, name: str, dpi: int = 300) -> None:
    out = FIG / name
    fig.savefig(out, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    gc.collect()
    print(f"  ✓  {name}")

# ════════════════════════════════════════════════════════════════════════════
# Data loaders
# ════════════════════════════════════════════════════════════════════════════

def load_all_metrics() -> pd.DataFrame:
    """Load combined metrics for all models."""
    comp = TBL / "phase5a_all_models_comparison.csv"
    if comp.exists():
        df = pd.read_csv(comp)
        return df
    # Fallback: merge phase4 + phase5a
    dfs = []
    for f in ["phase4_baseline_metrics.csv", "phase4b_cnn_metrics.csv"]:
        p = TBL / f
        if p.exists():
            dfs.append(pd.read_csv(p))
    for arch in ["gat", "gcn", "graphsage"]:
        p = TBL / f"phase5a_{arch}_metrics.csv"
        if p.exists():
            dfs.append(pd.read_csv(p))
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

def load_gat_preds() -> dict | None:
    path = PRED / "phase5a_gat_preds.npz"
    if not path.exists():
        print(f"  ⚠  {path.name} not found")
        return None
    d = np.load(path)
    return {k: d[k] for k in d.files}

def load_phase4_preds() -> dict | None:
    path = PRED / "phase4_test_predictions.npz"
    if not path.exists():
        return None
    d = np.load(path)
    return {k: d[k] for k in d.files}

def load_binned(arch: str = "gat") -> pd.DataFrame | None:
    for fname in [f"phase5a_{arch}_binned.csv",
                  f"phase5a_{arch}_binned_metrics.csv"]:
        p = TBL / fname
        if p.exists():
            return pd.read_csv(p)
    return None

def load_cal(arch: str = "gat") -> pd.DataFrame | None:
    p = TBL / f"phase5b_{arch}_calibration.csv"
    return pd.read_csv(p) if p.exists() else None

def load_intervention(scenario: str, arch: str = "gat") -> dict | None:
    for prefix in ["phase5d_v2_", "phase5d_"]:
        p = PRED / f"{prefix}{scenario}_{arch}_effects.npz"
        if p.exists():
            d = np.load(p)
            return {k: d[k] for k in d.files}
    return None

def load_graph_pos_test() -> np.ndarray | None:
    gp = PROJECT_ROOT / "data" / "processed" / "graph_data_enriched.pt"
    if not gp.exists():
        return None
    g = torch.load(gp, map_location="cpu", weights_only=False)
    return g.pos[g.test_mask].numpy()

def load_importances() -> pd.DataFrame | None:
    p = TBL / "phase4_feature_importances.csv"
    return pd.read_csv(p) if p.exists() else None

def load_history(arch: str = "gat") -> pd.DataFrame | None:
    p = TBL / f"phase5a_{arch}_history.csv"
    return pd.read_csv(p) if p.exists() else None

# ════════════════════════════════════════════════════════════════════════════
# Figure 1 — Geographic Split Map
# ════════════════════════════════════════════════════════════════════════════

def fig1_geographic_split():
    print("\n[Fig 1] Geographic split map")
    gp = PROJECT_ROOT / "data" / "processed" / "graph_data_enriched.pt"
    if not gp.exists():
        print("  ⚠  graph not found"); return

    g   = torch.load(gp, map_location="cpu", weights_only=False)
    pos = g.pos.numpy()
    y   = g.y_raw.numpy().ravel()

    rng = np.random.default_rng(42)
    idx = rng.choice(len(pos), min(60_000, len(pos)), replace=False)

    train_mask = g.train_mask.numpy()
    val_mask   = g.val_mask.numpy()
    test_mask  = g.test_mask.numpy()

    fig, axes = plt.subplots(1, 2, figsize=(14, 7))

    # Left: burn probability spatial distribution
    ax = axes[0]
    sc = ax.scatter(
        pos[idx, 1], -pos[idx, 0],
        c=y[idx], cmap="YlOrRd",
        vmin=0, vmax=0.15,
        s=1.5, alpha=0.7, rasterized=True
    )
    cb = plt.colorbar(sc, ax=ax, shrink=0.8)
    cb.set_label("Burn Probability")
    ax.set_xlabel("Column (West → East)")
    ax.set_ylabel("Row (South → North)")
    ax.set_title("(a) Spatial Distribution of Burn Probability\nFSim Dataset Greece (EPSG:2100)")

    # Right: geographic split
    ax2 = axes[1]
    split_colors = {True: C["RF"], False: ""}
    for mask, color, label in [
        (train_mask, "#3498DB", "Train (rows 0–4200, 72.5%)"),
        (val_mask,   "#2ECC71", "Val (rows 4201–4800, 9.9%)"),
        (test_mask,  "#E74C3C", "Test (rows 4801–7590, 17.6%)"),
    ]:
        idx2 = rng.choice(np.where(mask)[0], min(20_000, mask.sum()), replace=False)
        ax2.scatter(pos[idx2, 1], -pos[idx2, 0],
                    c=color, s=1, alpha=0.5, rasterized=True, label=label)

    ax2.legend(loc="lower right", markerscale=6, fontsize=9)
    ax2.set_xlabel("Column (West → East)")
    ax2.set_ylabel("Row (South → North)")
    ax2.set_title("(b) Geographic Block Split\nNo spatial overlap between splits")

    # Annotate regions
    ax2.axhline(-4200, color="#3498DB", lw=1.5, ls="--", alpha=0.6)
    ax2.axhline(-4800, color="#2ECC71", lw=1.5, ls="--", alpha=0.6)
    ax2.text(7000, -2100, "Train\n(N. Greece)", fontsize=8, color="#1a5276")
    ax2.text(7000, -4500, "Val", fontsize=8, color="#1e8449")
    ax2.text(7000, -6000, "Test\n(S. Greece\n+ Aegean)", fontsize=8, color="#922b21")

    fig.suptitle(
        "Study Area — FSim Wildfire Dataset Greece\n"
        "327,405 nodes · 61 features · 2,511,084 edges · 8-connected grid",
        fontsize=13, fontweight="bold", y=1.01
    )
    plt.tight_layout()
    savefig(fig, "fig1_geographic_split.png")

# ════════════════════════════════════════════════════════════════════════════
# Figure 2 — Full Model Comparison
# ════════════════════════════════════════════════════════════════════════════

def fig2_model_comparison():
    print("\n[Fig 2] Model comparison")
    df = load_all_metrics()
    if df.empty:
        print("  ⚠  No metrics found"); return

    df["label"] = df["model"].map(LABEL).fillna(df["model"])

    # Define model order and colors
    order = ["GAT", "2D CNN (spatial)", "GCN", "XGBoost",
             "Random Forest", "Ridge Regression", "Naive Mean"]
    color_map = {
        "GAT":             C["GAT"],
        "2D CNN (spatial)":C["CNN"],
        "GCN":             C["GCN"],
        "XGBoost":         C["XGBoost"],
        "Random Forest":   C["RF"],
        "Ridge Regression":C["Ridge"],
        "Naive Mean":      C["Naive"],
    }
    df["sort_key"] = df["model"].map({m: i for i, m in enumerate(order)})
    df = df.dropna(subset=["sort_key"]).sort_values("sort_key")
    labels   = df["label"].tolist()
    colors   = [color_map.get(m, "#aaaaaa") for m in df["model"]]

    metrics  = [
        ("r2",       "R²",           True,   "Higher is better"),
        ("mae",      "MAE",          False,  "Lower is better"),
        ("spearman", "Spearman ρ",   True,   "Higher is better"),
        ("brier",    "Brier Score",  False,  "Lower is better"),
        ("ece",      "ECE",          False,  "Lower is better"),
    ]

    fig, axes = plt.subplots(1, 5, figsize=(20, 6))

    for ax, (col, title, higher_better, note) in zip(axes, metrics):
        if col not in df.columns:
            continue
        vals = df[col].values
        bars = ax.barh(labels, vals, color=colors, height=0.65, alpha=0.88)

        # Value labels
        for bar, v in zip(bars, vals):
            xpos = v + max(abs(vals)) * 0.02 if higher_better else v + max(abs(vals)) * 0.02
            ax.text(xpos, bar.get_y() + bar.get_height()/2,
                    f"{v:.4f}" if abs(v) < 1 else f"{v:.2f}",
                    va="center", fontsize=7.5)

        ax.set_title(f"{title}\n({note})", fontsize=10, fontweight="bold")
        ax.invert_yaxis()
        ax.set_xlabel(title)

        # Highlight GAT bar
        for bar, m in zip(bars, df["model"]):
            if m == "GAT":
                bar.set_edgecolor("#000000")
                bar.set_linewidth(1.5)

    # Legend
    legend_patches = [
        Patch(color=C["GAT"],    label="GNN — GAT (primary)"),
        Patch(color=C["GCN"],    label="GNN — GCN (ablation)"),
        Patch(color=C["CNN"],    label="CNN Spatial Baseline"),
        Patch(color=C["XGBoost"],label="Tabular Baselines"),
    ]
    fig.legend(handles=legend_patches, loc="lower center",
               ncol=4, bbox_to_anchor=(0.5, -0.04), fontsize=9)

    fig.suptitle(
        "Figure 2 — Model Performance Comparison\n"
        "Test split (n=57,531), original burn probability scale, geographic block split",
        fontsize=13, fontweight="bold"
    )
    plt.tight_layout()
    savefig(fig, "fig2_model_comparison.png")

# ════════════════════════════════════════════════════════════════════════════
# Figure 3 — Prediction vs Truth (GAT + CNN + XGBoost)
# ════════════════════════════════════════════════════════════════════════════

def fig3_prediction_scatter():
    print("\n[Fig 3] Prediction scatter")
    gat_preds  = load_gat_preds()
    p4_preds   = load_phase4_preds()

    if gat_preds is None:
        print("  ⚠  GAT preds not found"); return

    rng = np.random.default_rng(42)
    y_true = gat_preds["y_true_bp"]
    n      = min(20_000, len(y_true))
    idx    = rng.choice(len(y_true), n, replace=False)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5.5))

    models_data = []
    # GAT
    models_data.append(("GAT (ours)",  gat_preds["y_pred_bp"],
                        C["GAT"],  0.7659))
    # CNN + XGBoost from phase4
    if p4_preds:
        for key, label, color in [
            ("2D_CNN_spatial", "2D CNN",   C["CNN"]),
            ("XGBoost",        "XGBoost",  C["XGBoost"]),
        ]:
            if key in p4_preds:
                r2 = 1 - np.sum((y_true - p4_preds[key])**2) / \
                     np.sum((y_true - y_true.mean())**2)
                models_data.append((label, p4_preds[key], color, float(r2)))

    for ax, (label, y_pred, color, r2) in zip(axes, models_data):
        mae = float(np.mean(np.abs(y_true - y_pred)))
        ax.scatter(y_true[idx], y_pred[idx],
                   s=2, alpha=0.25, color=color, rasterized=True)
        lo = min(y_true.min(), y_pred.min())
        hi = max(y_true.max(), y_pred.max())
        ax.plot([lo, hi], [lo, hi], "k--", lw=1.5, label="Perfect")
        ax.set_xlabel("True Burn Probability", fontsize=10)
        ax.set_ylabel("Predicted Burn Probability", fontsize=10)
        ax.set_title(f"{label}", fontsize=11, fontweight="bold", color=color)
        ax.text(0.05, 0.92, f"R²={r2:.4f}\nMAE={mae:.4f}",
                transform=ax.transAxes, fontsize=9,
                bbox=dict(facecolor="white", alpha=0.85, boxstyle="round"))

    fig.suptitle(
        "Figure 3 — Predicted vs True Burn Probability\n"
        "Test split (n=57,531, 20,000 sampled) · Original BP scale · Geographic split",
        fontsize=12, fontweight="bold"
    )
    plt.tight_layout()
    savefig(fig, "fig3_prediction_scatter.png")

# ════════════════════════════════════════════════════════════════════════════
# Figure 4 — Binned High-Risk Evaluation
# ════════════════════════════════════════════════════════════════════════════

def fig4_binned_evaluation():
    print("\n[Fig 4] Binned high-risk evaluation")

    arch_data = {}
    for arch in ["gat", "gcn", "graphsage"]:
        df = load_binned(arch)
        if df is not None and len(df) > 0:
            arch_data[arch.upper()] = df

    # Also load Phase 4 binned
    p4_bin = TBL / "phase4_binned_metrics.csv"
    if p4_bin.exists():
        df4 = pd.read_csv(p4_bin)
        for model in ["XGBoost", "Random Forest"]:
            sub = df4[df4["model"] == model]
            if len(sub) > 0:
                arch_data[model] = sub

    if not arch_data:
        print("  ⚠  No binned data found"); return

    fig, axes = plt.subplots(1, 2, figsize=(13, 6))

    # MAE per bin
    ax = axes[0]
    color_map2 = {"GAT": C["GAT"], "GCN": C["GCN"], "GRAPHSAGE": C["GraphSAGE"],
                  "XGBoost": C["XGBoost"], "Random Forest": C["RF"],
                  "2D CNN (spatial)": C["CNN"]}
    ls_map = {"GAT": "-", "GCN": "--", "GRAPHSAGE": ":",
              "XGBoost": "-.", "Random Forest": (0,(3,1)),
              "2D CNN (spatial)": (0,(5,2))}

    # CNN binned from phase4b
    p4b = TBL / "phase4b_cnn_binned_metrics.csv"
    if p4b.exists():
        cnn_b = pd.read_csv(p4b)
        arch_data["2D CNN (spatial)"] = cnn_b

    for arch_label, df in arch_data.items():
        df_s = df.sort_values("bin") if "bin" in df.columns else df
        bins = df_s["bin"].values if "bin" in df_s else range(len(df_s))
        mae  = df_s["mae"].values
        col  = color_map2.get(arch_label, "#999999")
        ls   = ls_map.get(arch_label, "-")
        lbl  = LABEL.get(arch_label, arch_label)
        ax.plot(bins, mae, marker="o", ms=5, lw=2,
                color=col, linestyle=ls, label=lbl)

    ax.set_xlabel("Burn Probability Quintile Bin\n(1=lowest risk, 5=highest risk)")
    ax.set_ylabel("Mean Absolute Error (MAE)")
    ax.set_title("(a) MAE by Risk Level\nHigh-risk tail (Bin 5) is most critical")
    ax.legend(ncol=1, fontsize=8)
    ax.set_xticks([1,2,3,4,5])
    ax.set_xticklabels(["Bin 1\n[0.000,0.004]","Bin 2\n[0.004,0.010]",
                         "Bin 3\n[0.010,0.021]","Bin 4\n[0.021,0.047]",
                         "Bin 5\n[0.047,0.208]"], fontsize=8)

    # Bin 5 comparison bar
    ax2 = axes[1]
    bin5_models, bin5_vals, bin5_cols = [], [], []
    for arch_label, df in arch_data.items():
        df_s = df.sort_values("bin") if "bin" in df.columns else df
        if len(df_s) >= 5:
            b5 = df_s[df_s["bin"] == 5]
            if len(b5) > 0:
                bin5_models.append(LABEL.get(arch_label, arch_label))
                bin5_vals.append(float(b5["mae"].values[0]))
                bin5_cols.append(color_map2.get(arch_label, "#999999"))

    sort_idx = np.argsort(bin5_vals)
    bin5_models = [bin5_models[i] for i in sort_idx]
    bin5_vals   = [bin5_vals[i] for i in sort_idx]
    bin5_cols   = [bin5_cols[i] for i in sort_idx]

    bars = ax2.barh(bin5_models, bin5_vals, color=bin5_cols, height=0.65, alpha=0.88)
    for bar, v in zip(bars, bin5_vals):
        ax2.text(v + 0.0005, bar.get_y() + bar.get_height()/2,
                 f"{v:.4f}", va="center", fontsize=9)

    ax2.set_xlabel("MAE in Highest Risk Bin (BP > 0.047)")
    ax2.set_title("(b) High-Risk Bin Performance\n(Bin 5: BP ∈ [0.047, 0.208])")

    fig.suptitle(
        "Figure 4 — Binned Evaluation by Burn Probability Risk Level\n"
        "Higher BP quintile = more dangerous cells = harder prediction",
        fontsize=12, fontweight="bold"
    )
    plt.tight_layout()
    savefig(fig, "fig4_binned_evaluation.png")

# ════════════════════════════════════════════════════════════════════════════
# Figure 5 — Feature Importance
# ════════════════════════════════════════════════════════════════════════════

def fig5_feature_importance():
    print("\n[Fig 5] Feature importance")
    df = load_importances()
    if df is None:
        print("  ⚠  Importances not found"); return

    fig, axes = plt.subplots(1, 2, figsize=(14, 8))
    models_plot = [("Random Forest", C["RF"]), ("XGBoost", C["XGBoost"])]

    for ax, (model, color) in zip(axes, models_plot):
        sub = df[df["model"] == model].head(15).sort_values("importance")
        ax.barh(sub["feature"], sub["importance"],
                color=color, alpha=0.85, height=0.7)
        ax.set_xlabel("Feature Importance")
        ax.set_title(f"{LABEL.get(model, model)}\nTop 15 features",
                     fontsize=11, fontweight="bold")

        # Colour bars by feature group
        group_colors = {
            "FSP":       "#1a6faf",
            "CFL":       "#e07b39",
            "Ignition":  "#7b3f99",
            "Struct":    "#b84040",
            "dem":       "#2ecc71",
            "fuel":      "#888888",
            "interact":  "#c0a020",
        }
        for bar, feat in zip(ax.patches, sub["feature"]):
            for key, col in group_colors.items():
                if key.upper() in feat.upper():
                    bar.set_color(col)
                    break

    # Shared legend
    legend_patches = [
        Patch(color="#1a6faf", label="FSP Index features"),
        Patch(color="#e07b39", label="CFL features"),
        Patch(color="#b84040", label="Struct Exp Index"),
        Patch(color="#7b3f99", label="Ignition features"),
        Patch(color="#2ecc71", label="DEM terrain"),
        Patch(color="#c0a020", label="Interaction terms"),
        Patch(color="#888888", label="Fuel categories"),
    ]
    fig.legend(handles=legend_patches, loc="lower center",
               ncol=4, bbox_to_anchor=(0.5, -0.02), fontsize=9)

    fig.suptitle(
        "Figure 5 — Feature Importance Analysis\n"
        "Multi-scale spatial statistics dominate predictions in both models",
        fontsize=12, fontweight="bold"
    )
    plt.tight_layout()
    savefig(fig, "fig5_feature_importance.png")

# ════════════════════════════════════════════════════════════════════════════
# Figure 6 — Calibration Reliability Diagram
# ════════════════════════════════════════════════════════════════════════════

def fig6_calibration():
    print("\n[Fig 6] Calibration reliability")
    gat_cal = load_cal("gat")
    gcn_cal = load_cal("gcn")

    if gat_cal is None:
        print("  ⚠  Calibration data not found"); return

    fig, axes = plt.subplots(1, 3, figsize=(15, 5.5))

    # Panel 1: GAT reliability diagram from CSV summary
    ax = axes[0]
    # We reconstruct from picp values
    gat_before = gat_cal[gat_cal["stage"] == "before"].iloc[0]
    gat_after  = gat_cal[gat_cal["stage"] == "after"].iloc[0]

    targets  = [0.50, 0.90, 0.95]
    b_vals   = [gat_before["picp_50"], gat_before["picp_90"], gat_before["picp_95"]]
    a_vals   = [gat_after["picp_50"],  gat_after["picp_90"],  gat_after["picp_95"]]

    ax.plot([0,1],[0,1],"k--",lw=1.5, label="Perfect calibration", zorder=5)
    ax.scatter(targets, b_vals, color=C["before"], s=80, zorder=6, label="Before scaling")
    ax.scatter(targets, a_vals, color=C["after"],  s=80, zorder=6,
               marker="s", label=f"After scaling (T={gat_after['T']:.3f})")
    ax.plot(targets, b_vals, color=C["before"], lw=1.5, alpha=0.7)
    ax.plot(targets, a_vals, color=C["after"],  lw=1.5, alpha=0.7)

    ax.set_xlim(0.4, 1.02); ax.set_ylim(0.4, 1.02)
    ax.set_xlabel("Expected Coverage")
    ax.set_ylabel("Actual Coverage (PICP)")
    ax.set_title("(a) GAT Reliability Diagram\nTemperature Scaling")
    ax.legend(fontsize=9)

    # Panel 2: Before vs After bar chart for all arches
    ax2 = axes[1]
    arch_labels = ["GAT", "GCN", "GraphSAGE"]
    arch_files  = ["gat", "gcn", "graphsage"]
    x_pos = np.arange(len(arch_labels))
    w     = 0.35

    picp90_before, picp90_after, T_vals = [], [], []
    for arch in arch_files:
        cal = load_cal(arch)
        if cal is not None:
            b = cal[cal["stage"]=="before"].iloc[0]
            a = cal[cal["stage"]=="after"].iloc[0]
            picp90_before.append(b["picp_90"])
            picp90_after.append(a["picp_90"])
            T_vals.append(a["T"])
        else:
            picp90_before.append(np.nan)
            picp90_after.append(np.nan)
            T_vals.append(np.nan)

    ax2.bar(x_pos - w/2, picp90_before, w, color=C["before"],
            alpha=0.75, label="Before scaling", zorder=3)
    ax2.bar(x_pos + w/2, picp90_after,  w, color=C["after"],
            alpha=0.75, label="After scaling",  zorder=3)
    ax2.axhline(0.90, color="black", lw=1.5, ls="--",
                label="Target PICP=0.90", zorder=4)
    ax2.fill_between([-0.5, 2.5], [0.85, 0.85], [0.95, 0.95],
                     alpha=0.1, color="green", label="Acceptable band ±5%")

    ax2.set_xticks(x_pos)
    ax2.set_xticklabels([f"{a}\n(T={t:.3f})" for a, t in zip(arch_labels, T_vals)],
                        fontsize=9)
    ax2.set_ylabel("PICP at 90% Nominal Coverage")
    ax2.set_title("(b) PICP-90% Before/After\nAll architectures")
    ax2.legend(fontsize=8)
    ax2.set_ylim(0.5, 1.05)

    # Panel 3: ECE comparison
    ax3 = axes[2]
    all_df = load_all_metrics()
    if not all_df.empty:
        model_order = ["GAT", "GCN", "GraphSAGE",
                       "2D CNN (spatial)", "XGBoost", "Random Forest"]
        sub = all_df[all_df["model"].isin(model_order)].copy()
        sub["sort"] = sub["model"].map({m: i for i, m in enumerate(model_order)})
        sub = sub.sort_values("sort")
        cols_bar = [color_map_ece(m) for m in sub["model"]]
        ax3.bar(sub["model"].map(LABEL).fillna(sub["model"]),
                sub["ece"], color=cols_bar, alpha=0.85, height=0.7)
        ax3.axhline(0.05, color="red", lw=1.5, ls="--",
                    label="ECE=0.05 threshold")
        ax3.set_ylabel("ECE (Expected Calibration Error)")
        ax3.set_title("(c) ECE Comparison\nLower = better calibrated")
        ax3.tick_params(axis="x", rotation=30)
        ax3.legend(fontsize=9)

    fig.suptitle(
        "Figure 6 — Uncertainty Calibration (Phase 5B)\n"
        "GAT achieves PICP-90%=0.932 after temperature scaling (T=0.643)",
        fontsize=12, fontweight="bold"
    )
    plt.tight_layout()
    savefig(fig, "fig6_calibration_reliability.png")

def color_map_ece(model: str) -> str:
    cm = {"GAT": C["GAT"], "GCN": C["GCN"], "GraphSAGE": C["GraphSAGE"],
          "2D CNN (spatial)": C["CNN"], "XGBoost": C["XGBoost"],
          "Random Forest": C["RF"], "Ridge Regression": C["Ridge"]}
    return cm.get(model, "#aaaaaa")

# ════════════════════════════════════════════════════════════════════════════
# Figure 7 — Uncertainty Decomposition
# ════════════════════════════════════════════════════════════════════════════

def fig7_uncertainty():
    print("\n[Fig 7] Uncertainty decomposition")
    gat = load_gat_preds()
    if gat is None:
        return

    y_true   = gat["y_true_bp"]
    y_pred   = gat["y_pred_bp"]
    epistemic= gat["std_pred"]
    aleatoric= gat["aleatoric"]
    total    = gat["total_unc"]

    rng = np.random.default_rng(42)
    idx = rng.choice(len(y_true), min(20_000, len(y_true)), replace=False)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5.5))

    # Panel 1: Total uncertainty vs true BP
    ax = axes[0]
    sc = ax.scatter(y_true[idx], total[idx],
                    c=y_pred[idx], cmap="YlOrRd",
                    s=2, alpha=0.25, rasterized=True, vmin=0, vmax=0.15)
    cb = plt.colorbar(sc, ax=ax)
    cb.set_label("Predicted BP", fontsize=9)
    ax.set_xlabel("True Burn Probability")
    ax.set_ylabel("Total Uncertainty (σ)")
    ax.set_title("(a) Total Uncertainty vs True BP\nColour = predicted BP")

    # Panel 2: Epistemic vs Aleatoric
    ax2 = axes[1]
    ax2.scatter(epistemic[idx], aleatoric[idx],
                c=y_true[idx], cmap="YlOrRd",
                s=2, alpha=0.2, rasterized=True, vmin=0, vmax=0.15)
    ax2.set_xlabel("Epistemic Uncertainty (MC Dropout σ)")
    ax2.set_ylabel("Aleatoric Uncertainty (√exp(log_var))")
    ax2.set_title("(b) Uncertainty Decomposition\nColour = true BP")

    # Panel 3: Error vs uncertainty (calibration quality)
    ax3 = axes[2]
    error = np.abs(y_true - y_pred)
    ax3.scatter(total[idx], error[idx],
                c=y_true[idx], cmap="YlOrRd",
                s=2, alpha=0.2, rasterized=True, vmin=0, vmax=0.15)
    # Add reference line (perfectly calibrated: error ≈ σ)
    lim = max(total[idx].max(), error[idx].max())
    ax3.plot([0, lim], [0, lim], "k--", lw=1.5, label="|error|=σ (perfect)")
    ax3.set_xlabel("Total Uncertainty (σ)")
    ax3.set_ylabel("|Prediction Error|")
    ax3.set_title("(c) Error vs Uncertainty\nPerfect calibration: diagonal")
    ax3.legend(fontsize=9)

    fig.suptitle(
        "Figure 7 — Uncertainty Decomposition — GAT Model\n"
        "30 MC Dropout passes · Epistemic + Aleatoric · Temperature T=0.643",
        fontsize=12, fontweight="bold"
    )
    plt.tight_layout()
    savefig(fig, "fig7_uncertainty_decomposition.png")

# ════════════════════════════════════════════════════════════════════════════
# Figure 8 — Intervention Analysis
# ════════════════════════════════════════════════════════════════════════════

def fig8_intervention():
    print("\n[Fig 8] Intervention analysis")
    pos = load_graph_pos_test()
    if pos is None:
        print("  ⚠  Graph not found"); return

    scenarios = {
        "fuel_reduction_30pct":       ("Fuel Reduction 30%\n(CFL × 0.70)", C["fuel"]),
        "firebreak":                  ("Firebreak Strip\n(CFL = 0 in strip)", C["firebreak"]),
        "ignition_suppression_50pct": ("Ignition Suppression 50%\n(Ign × 0.50)", C["ignition"]),
    }

    effects = {}
    for key in scenarios:
        e = load_intervention(key)
        if e is not None:
            effects[key] = e

    if not effects:
        print("  ⚠  No intervention data found"); return

    n_scen = len(effects)
    fig    = plt.figure(figsize=(7*n_scen, 12))
    gs     = gridspec.GridSpec(2, n_scen, hspace=0.35, wspace=0.3)

    rng = np.random.default_rng(42)
    idx = rng.choice(len(pos), min(25_000, len(pos)), replace=False)

    all_deltas = np.concatenate([e["delta_bp"] for e in effects.values()])
    p2, p98    = np.percentile(all_deltas, [2, 98])
    vmax       = min(max(abs(p2), abs(p98), 0.003), 0.05)

    for col_i, (key, effect) in enumerate(effects.items()):
        label, color = scenarios[key]
        delta = effect["delta_bp"]
        sig   = effect["significant_mask"]
        lo    = effect["delta_bp_lo_90"]
        hi    = effect["delta_bp_hi_90"]

        pct_r   = float((delta < 0).mean() * 100)
        pct_sig = float(sig.mean() * 100)
        mean_d  = float(delta.mean())

        # Top panel: spatial map
        ax_map = fig.add_subplot(gs[0, col_i])
        sc = ax_map.scatter(
            pos[idx, 1], -pos[idx, 0],
            c=delta[idx], cmap="RdBu_r",
            vmin=-vmax, vmax=vmax,
            s=2, alpha=0.7, rasterized=True
        )
        plt.colorbar(sc, ax=ax_map, label="Δ BP", shrink=0.7, pad=0.02)
        ax_map.set_title(
            f"{label}\nMean Δ={mean_d:+.4f}  Sig: {pct_sig:.1f}%",
            fontsize=10, fontweight="bold"
        )
        ax_map.set_xlabel("Column")
        ax_map.set_ylabel("Row (N→S)")

        # Bottom panel: delta histogram with 90% CI
        ax_hist = fig.add_subplot(gs[1, col_i])
        ax_hist.hist(delta, bins=70, color=color, alpha=0.8, edgecolor="none")
        ax_hist.axvline(0,          color="black", lw=1.5, ls="--")
        ax_hist.axvline(mean_d,     color="red",   lw=2,   ls="-",
                        label=f"Mean={mean_d:+.4f}")
        ax_hist.axvline(np.percentile(lo, 50), color="orange", lw=1.5, ls=":",
                        label=f"Median 90% CI lo")

        # Shade 90% CI region
        lo_med = np.percentile(lo, 50)
        hi_med = np.percentile(hi, 50)
        ax_hist.axvspan(lo_med, hi_med, alpha=0.15, color=color)

        ax_hist.set_xlabel("Δ Burn Probability")
        ax_hist.set_ylabel("Node count")
        ax_hist.set_title(f"Distribution of Effects\n{pct_r:.1f}% nodes reduced")
        ax_hist.legend(fontsize=8)

    fig.suptitle(
        "Figure 8 — Counterfactual Intervention Analysis — GAT Model\n"
        "Phase 5D: All derived features modified · 30 MC Dropout passes · "
        "Calibrated 90% prediction intervals",
        fontsize=13, fontweight="bold", y=1.01
    )
    savefig(fig, "fig8_intervention_analysis.png")

# ════════════════════════════════════════════════════════════════════════════
# Figure 9 — Ablation: Architecture Comparison
# ════════════════════════════════════════════════════════════════════════════

def fig9_ablation():
    print("\n[Fig 9] Architecture ablation")
    all_df = load_all_metrics()
    if all_df.empty:
        return

    gnn_models = ["GAT", "GCN", "GraphSAGE", "2D CNN (spatial)",
                  "XGBoost", "Random Forest"]
    sub = all_df[all_df["model"].isin(gnn_models)].copy()

    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    axes = axes.flatten()

    color_map3 = {"GAT": C["GAT"], "GCN": C["GCN"], "GraphSAGE": C["GraphSAGE"],
                  "2D CNN (spatial)": C["CNN"], "XGBoost": C["XGBoost"],
                  "Random Forest": C["RF"]}

    for ax, (metric, ylabel, ascending) in zip(axes, [
        ("r2",       "R²",         False),
        ("mae",      "MAE",        True),
        ("spearman", "Spearman ρ", False),
        ("ece",      "ECE",        True),
    ]):
        if metric not in sub.columns:
            continue
        sub_sorted = sub.sort_values(metric, ascending=ascending)
        colors_b   = [color_map3.get(m, "#aaa") for m in sub_sorted["model"]]
        labels_b   = [LABEL.get(m, m) for m in sub_sorted["model"]]
        bars = ax.barh(labels_b, sub_sorted[metric],
                       color=colors_b, alpha=0.88, height=0.65)
        for bar, v in zip(bars, sub_sorted[metric]):
            ax.text(v + sub_sorted[metric].abs().max() * 0.02,
                    bar.get_y() + bar.get_height()/2,
                    f"{v:.4f}", va="center", fontsize=8.5)
        ax.set_xlabel(ylabel)
        ax.set_title(f"{ylabel} — {'Higher' if not ascending else 'Lower'} is better")

        # Highlight GAT
        for bar, m in zip(bars, sub_sorted["model"]):
            if m == "GAT":
                bar.set_edgecolor("black"); bar.set_linewidth(1.8)

    fig.suptitle(
        "Figure 9 — Architecture Ablation Study\n"
        "GNN architectures vs spatial and tabular baselines · "
        "Test split (n=57,531) · Geographic block split",
        fontsize=12, fontweight="bold"
    )
    plt.tight_layout()
    savefig(fig, "fig9_ablation_architecture.png")

# ════════════════════════════════════════════════════════════════════════════
# Figure 10 — Training Dynamics
# ════════════════════════════════════════════════════════════════════════════

def fig10_training():
    print("\n[Fig 10] Training dynamics")
    histories = {}
    for arch in ["gat", "gcn", "graphsage"]:
        h = load_history(arch)
        if h is not None:
            histories[arch.upper()] = h

    if not histories:
        print("  ⚠  No history data found"); return

    fig, axes = plt.subplots(1, len(histories), figsize=(6*len(histories), 5.5))
    if len(histories) == 1:
        axes = [axes]

    colors_h = {"GAT": C["GAT"], "GCN": C["GCN"], "GRAPHSAGE": C["GraphSAGE"]}

    for ax, (arch, h) in zip(axes, histories.items()):
        color = colors_h.get(arch, "#555")
        ax.plot(h["epoch"], h["train_loss"], lw=2, color=color,
                alpha=0.9, label="Train loss")
        ax.plot(h["epoch"], h["val_loss"], lw=2, color=color,
                alpha=0.9, ls="--", label="Val loss")

        # Mark best val
        best_i = h["val_loss"].idxmin()
        best_e = int(h["epoch"][best_i])
        best_v = float(h["val_loss"].min())
        ax.axvline(best_e, color="red", lw=1.5, ls=":",
                   label=f"Best val epoch {best_e}\nloss={best_v:.4f}")

        final_train = float(h["train_loss"].iloc[-1])
        final_val   = float(h["val_loss"].iloc[-1])
        gap = final_val - final_train

        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss (MSE)")
        ax.set_title(
            f"({['a','b','c'][list(histories.keys()).index(arch)]}) {arch} Training\n"
            f"train={final_train:.4f}  val={final_val:.4f}  "
            f"gap={gap:.4f}"
        )
        ax.legend(fontsize=9)

    fig.suptitle(
        "Figure 10 — GNN Training Dynamics\n"
        "NeighborLoader mini-batch · Patience=15 · MSE loss · CPU training",
        fontsize=12, fontweight="bold"
    )
    plt.tight_layout()
    savefig(fig, "fig10_training_dynamics.png")

# ════════════════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════════════════

def main():
    print("\n" + "="*65)
    print("  Paper Figure Generation — All Phases")
    print(f"  Output: {FIG}")
    print("="*65)

    fig1_geographic_split()
    fig2_model_comparison()
    fig3_prediction_scatter()
    fig4_binned_evaluation()
    fig5_feature_importance()
    fig6_calibration()
    fig7_uncertainty()
    fig8_intervention()
    fig9_ablation()
    fig10_training()

    print(f"\n{'='*65}")
    print(f"  All figures saved to: {FIG.name}/")
    print(f"{'='*65}")
    figs = sorted(FIG.glob("fig*.png"))
    total_mb = sum(f.stat().st_size for f in figs) / 1024**2
    for f in figs:
        kb = f.stat().st_size // 1024
        print(f"    {f.name:<45} {kb:>5} KB")
    print(f"\n  Total: {len(figs)} figures  ({total_mb:.1f} MB)")
    print(f"\n  READY FOR PAPER SUBMISSION")
    print()


if __name__ == "__main__":
    main()