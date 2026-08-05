"""
Publication-Quality Figures — SEPARATED (one plot per PNG)
===========================================================

Key differences from paper_figures.py:
  * Every subplot is saved as its OWN .png with a unique descriptive name
    (ready for \\includegraphics in Overleaf, one figure per file).
  

USAGE
-----
    conda activate wildfire-gnn
    cd spatiotemporal_wildfire_gnn
    python scripts/paper_figures2.py

Files that need per-node arrays (.npz) — fig1, fig3, fig7, fig8 — are only
produced if the corresponding .npz / graph file is present; otherwise the
script prints a clear SKIP note and continues.
"""

from __future__ import annotations
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sys, gc
import numpy as np
import pandas as pd
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

try:
    import torch
    HAS_TORCH = True
except Exception:
    HAS_TORCH = False

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# ── Paths ──────────────────────────────────────────────────────────────────
TBL  = PROJECT_ROOT / "reports" / "tables"
PRED = PROJECT_ROOT / "reports" / "predictions"
DATA = PROJECT_ROOT / "data" / "processed"
FIG  = PROJECT_ROOT / "reports" / "paper_figures2"      # <-- NEW folder
FIG.mkdir(parents=True, exist_ok=True)

# ── Style ──────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":       "DejaVu Sans",
    "font.size":         11,
    "axes.titlesize":    12,
    "axes.labelsize":    11,
    "xtick.labelsize":   10,
    "ytick.labelsize":   10,
    "legend.fontsize":   9,
    "figure.dpi":        150,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.alpha":        0.25,
    "grid.linestyle":    "--",
})

# ── Colour palette (family-based, consistent across all figures) ────────────
C = {
    "GAT":       "#1a6faf",   # deep blue   (primary GAT)
    "GATv2":     "#17becf",   # cyan
    "GAT_van":   "#8fd6df",   # light cyan
    "GCN":       "#4cae4f",   # green
    "GraphSAGE": "#7b3f99",   # purple
    "CNN":       "#e07b39",   # orange
    "XGBoost":   "#b84040",   # red
    "RF":        "#c0a020",   # gold
    "Ridge":     "#888888",   # grey
    "Naive":     "#cccccc",   # light grey
    "before":    "#e74c3c",
    "after":     "#2ecc71",
    "fuel":      "#e07b39",
    "firebreak": "#1a6faf",
    "ignition":  "#7b3f99",
}

# Display labels — NOTE: no "(ours)" / "(primary)" anywhere
LABEL = {
    "GAT":              "GAT",
    "GATv2":            "GATv2",
    "GATv2_vanilla":    "GATv2",
    "GAT_vanilla":      "GAT (vanilla)",
    "GCN":              "GCN",
    "GraphSAGE":        "GraphSAGE",
    "2D CNN (spatial)": "2D CNN",
    "XGBoost":          "XGBoost",
    "Random Forest":    "Random Forest",
    "Ridge Regression": "Ridge",
    "Ridge":            "Ridge",
    "Naive Mean":       "Naive Mean",
}

COLOR_FOR = {
    "GAT":              C["GAT"],
    "GATv2":            C["GATv2"],
    "GATv2_vanilla":    C["GATv2"],
    "GAT_vanilla":      C["GAT_van"],
    "GCN":              C["GCN"],
    "GraphSAGE":        C["GraphSAGE"],
    "2D CNN (spatial)": C["CNN"],
    "XGBoost":          C["XGBoost"],
    "Random Forest":    C["RF"],
    "Ridge Regression": C["Ridge"],
    "Ridge":            C["Ridge"],
    "Naive Mean":       C["Naive"],
}


def save(fig, name: str, dpi: int = 300) -> None:
    """Save one single-plot figure to the paper_figures2 folder."""
    out = FIG / name
    fig.savefig(out, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    gc.collect()
    print(f"  saved  {name}")


def _skip(name: str, reason: str) -> None:
    print(f"  SKIP   {name}  ({reason})")


# ════════════════════════════════════════════════════════════════════════════
# Loaders
# ════════════════════════════════════════════════════════════════════════════

def load_full_comparison() -> pd.DataFrame:
    """
    Full model comparison table. Start from phase5a_all_models_comparison.csv
    and merge in the phase5c GATv2 + GAT-vanilla rows so every model appears.
    """
    base = TBL / "phase5a_all_models_comparison.csv"
    frames = []
    if base.exists():
        frames.append(pd.read_csv(base))

    # Add GATv2 / GAT vanilla from phase5c individual metric files
    extra_specs = [
        ("phase5c_gatv2_metrics.csv",       "GATv2"),
        ("phase5c_gat_vanilla_metrics.csv", "GAT_vanilla"),
    ]
    for fname, disp in extra_specs:
        p = TBL / fname
        if p.exists():
            d = pd.read_csv(p)
            d = d.rename(columns={"model": "model"})
            d["model"] = disp     # normalise the model name for labelling
            keep = [c for c in ["model", "r2", "mae", "spearman", "brier", "ece"] if c in d.columns]
            frames.append(d[keep])

    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    # De-duplicate by model, keep first occurrence
    df = df.drop_duplicates(subset="model", keep="first").reset_index(drop=True)
    return df


def load_npz(path: Path) -> dict | None:
    if not path.exists():
        return None
    d = np.load(path)
    return {k: d[k] for k in d.files}


def load_binned(arch_file: str) -> pd.DataFrame | None:
    for fname in [f"phase5a_{arch_file}_binned.csv",
                  f"phase5c_{arch_file}_binned.csv",
                  f"phase5a_{arch_file}_binned_metrics.csv"]:
        p = TBL / fname
        if p.exists():
            return pd.read_csv(p)
    return None


def load_history(arch_file: str) -> pd.DataFrame | None:
    for fname in [f"phase5a_{arch_file}_history.csv",
                  f"phase5c_{arch_file}_history.csv"]:
        p = TBL / fname
        if p.exists():
            return pd.read_csv(p)
    return None


def load_importances() -> pd.DataFrame | None:
    p = TBL / "phase4_feature_importances.csv"
    return pd.read_csv(p) if p.exists() else None


# ════════════════════════════════════════════════════════════════════════════
# FIGURE 1 — Geographic split (2 separate PNGs) — needs graph .pt
# ════════════════════════════════════════════════════════════════════════════

def fig1_geographic_split():
    print("\n[Fig 1] Geographic split (needs graph_data_enriched.pt)")
    gp = DATA / "graph_data_enriched.pt"
    if not HAS_TORCH or not gp.exists():
        _skip("fig1a / fig1b", "graph_data_enriched.pt or torch unavailable")
        return

    g   = torch.load(gp, map_location="cpu", weights_only=False)
    pos = g.pos.numpy()
    y   = g.y_raw.numpy().ravel()
    rng = np.random.default_rng(42)
    idx = rng.choice(len(pos), min(60_000, len(pos)), replace=False)
    train_mask = g.train_mask.numpy()
    val_mask   = g.val_mask.numpy()
    test_mask  = g.test_mask.numpy()

    # ---- fig1a: burn probability spatial map ----
    fig, ax = plt.subplots(figsize=(7.5, 7))
    sc = ax.scatter(pos[idx, 1], -pos[idx, 0], c=y[idx], cmap="YlOrRd",
                    vmin=0, vmax=0.15, s=1.5, alpha=0.7, rasterized=True)
    cb = plt.colorbar(sc, ax=ax, shrink=0.8); cb.set_label("Burn Probability")
    ax.set_xlabel("Column (West → East)")
    ax.set_ylabel("Row (South → North)")
    ax.set_title("Spatial Distribution of Burn Probability\nFSim Dataset Greece (EPSG:2100)")
    plt.tight_layout(); save(fig, "fig1a_burn_probability_map.png")

    # ---- fig1b: geographic block split ----
    fig, ax = plt.subplots(figsize=(7.5, 7))
    for mask, color, label in [
        (train_mask, "#3498DB", "Train (rows 0–4200, 72.5%)"),
        (val_mask,   "#2ECC71", "Val (rows 4201–4800, 9.9%)"),
        (test_mask,  "#E74C3C", "Test (rows 4801–7590, 17.6%)"),
    ]:
        idx2 = rng.choice(np.where(mask)[0], min(20_000, mask.sum()), replace=False)
        ax.scatter(pos[idx2, 1], -pos[idx2, 0], c=color, s=1, alpha=0.5,
                   rasterized=True, label=label)
    ax.axhline(-4200, color="#3498DB", lw=1.5, ls="--", alpha=0.6)
    ax.axhline(-4800, color="#2ECC71", lw=1.5, ls="--", alpha=0.6)
    ax.legend(loc="lower right", markerscale=6, fontsize=9)
    ax.set_xlabel("Column (West → East)")
    ax.set_ylabel("Row (South → North)")
    ax.set_title("Geographic Block Split\nNo spatial overlap between splits")
    plt.tight_layout(); save(fig, "fig1b_geographic_split.png")


# ════════════════════════════════════════════════════════════════════════════
# FIGURE 2 — Model comparison (5 metrics -> 5 separate PNGs)
# ════════════════════════════════════════════════════════════════════════════

def _order_models(df: pd.DataFrame) -> pd.DataFrame:
    order = ["GAT", "2D CNN (spatial)", "GCN", "GATv2", "XGBoost",
             "Random Forest", "GAT_vanilla", "GraphSAGE",
             "Ridge Regression", "Naive Mean"]
    df = df.copy()
    df["sort_key"] = df["model"].map({m: i for i, m in enumerate(order)})
    df = df.dropna(subset=["sort_key"]).sort_values("sort_key")
    return df


def fig2_model_comparison():
    print("\n[Fig 2] Model comparison — 5 separate metric PNGs")
    df = load_full_comparison()
    if df.empty:
        _skip("fig2*", "no comparison table found"); return

    metrics = [
        ("r2",       "R²",          True,  "fig2a_r2.png"),
        ("mae",      "MAE",         False, "fig2b_mae.png"),
        ("spearman", "Spearman ρ",  True,  "fig2c_spearman.png"),
        ("brier",    "Brier Score", False, "fig2d_brier.png"),
        ("ece",      "ECE",         False, "fig2e_ece.png"),
    ]

    for col, title, higher_better, fname in metrics:
        if col not in df.columns:
            _skip(fname, f"column '{col}' missing"); continue
        d = _order_models(df)
        labels = [LABEL.get(m, m) for m in d["model"]]
        colors = [COLOR_FOR.get(m, "#aaaaaa") for m in d["model"]]
        vals   = d[col].values

        fig, ax = plt.subplots(figsize=(7, 6))
        bars = ax.barh(labels, vals, color=colors, height=0.68, alpha=0.9,
                       edgecolor="black", linewidth=0.5)
        # Highlight the primary GAT bar with a thicker border (no text label)
        for bar, m in zip(bars, d["model"]):
            if m == "GAT":
                bar.set_edgecolor("black"); bar.set_linewidth(2.2)
        span = max(abs(vals)) if len(vals) else 1.0
        for bar, v in zip(bars, vals):
            ax.text(v + span * 0.02, bar.get_y() + bar.get_height() / 2,
                    f"{v:.4f}", va="center", fontsize=8)
        note = "Higher is better" if higher_better else "Lower is better"
        ax.set_title(f"{title}\n({note})", fontweight="bold")
        ax.set_xlabel(title)
        ax.invert_yaxis()
        plt.tight_layout(); save(fig, fname)


# ════════════════════════════════════════════════════════════════════════════
# FIGURE 3 — Prediction scatter (GAT, XGBoost, CNN, GATv2 -> 4 separate PNGs)
# ════════════════════════════════════════════════════════════════════════════

def _r2(y_true, y_pred):
    return 1 - np.sum((y_true - y_pred) ** 2) / np.sum((y_true - y_true.mean()) ** 2)


def _scatter_png(y_true, y_pred, label, color, fname):
    rng = np.random.default_rng(42)
    n   = min(20_000, len(y_true))
    idx = rng.choice(len(y_true), n, replace=False)
    r2  = float(_r2(y_true, y_pred))
    mae = float(np.mean(np.abs(y_true - y_pred)))

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_true[idx], y_pred[idx], s=2, alpha=0.25, color=color, rasterized=True)
    lo = float(min(y_true.min(), y_pred.min()))
    hi = float(max(y_true.max(), y_pred.max()))
    ax.plot([lo, hi], [lo, hi], "k--", lw=1.5, label="Perfect")
    ax.set_xlabel("True Burn Probability")
    ax.set_ylabel("Predicted Burn Probability")
    ax.set_title(label, fontweight="bold", color=color)
    ax.text(0.05, 0.92, f"R²={r2:.4f}\nMAE={mae:.4f}", transform=ax.transAxes,
            fontsize=9, bbox=dict(facecolor="white", alpha=0.85, boxstyle="round"))
    plt.tight_layout(); save(fig, fname)


def fig3_prediction_scatter():
    print("\n[Fig 3] Prediction scatter — up to 4 separate PNGs")
    gat = load_npz(PRED / "phase5a_gat_preds.npz")
    p4  = load_npz(PRED / "phase4_test_predictions.npz")
    v2  = load_npz(PRED / "phase5c_gatv2_preds.npz")

    y_true = None
    if gat is not None and "y_true_bp" in gat:
        y_true = gat["y_true_bp"]

    # fig3a GAT
    if gat is not None and "y_pred_bp" in gat:
        _scatter_png(gat["y_true_bp"], gat["y_pred_bp"], "GAT", C["GAT"],
                     "fig3a_gat_scatter.png")
    else:
        _skip("fig3a_gat_scatter.png", "phase5a_gat_preds.npz not found")

    # fig3b XGBoost + fig3c CNN from phase4 preds
    if p4 is not None and y_true is not None:
        # try common key spellings
        xgb_key = next((k for k in p4 if k.lower().replace(" ", "") in
                        ("xgboost",)), None)
        cnn_key = next((k for k in p4 if "cnn" in k.lower()), None)
        if xgb_key:
            _scatter_png(y_true, p4[xgb_key], "XGBoost", C["XGBoost"],
                         "fig3b_xgboost_scatter.png")
        else:
            _skip("fig3b_xgboost_scatter.png", "XGBoost key not in phase4 npz")
        if cnn_key:
            _scatter_png(y_true, p4[cnn_key], "2D CNN", C["CNN"],
                         "fig3c_cnn_scatter.png")
        else:
            _skip("fig3c_cnn_scatter.png", "CNN key not in phase4 npz")
    else:
        _skip("fig3b / fig3c", "phase4_test_predictions.npz or y_true not found")

    # fig3d GATv2
    if v2 is not None:
        yt = v2["y_true_bp"] if "y_true_bp" in v2 else y_true
        yp_key = "y_pred_bp" if "y_pred_bp" in v2 else None
        if yt is not None and yp_key:
            _scatter_png(yt, v2[yp_key], "GATv2", C["GATv2"],
                         "fig3d_gatv2_scatter.png")
        else:
            _skip("fig3d_gatv2_scatter.png", "keys missing in phase5c_gatv2_preds.npz")
    else:
        _skip("fig3d_gatv2_scatter.png", "phase5c_gatv2_preds.npz not found")


# ════════════════════════════════════════════════════════════════════════════
# FIGURE 4 — Binned evaluation (2 separate PNGs: line + bin5 bar)
# ════════════════════════════════════════════════════════════════════════════

def fig4_binned_evaluation():
    print("\n[Fig 4] Binned evaluation — 2 separate PNGs")

    arch_specs = [
        ("gat",           "GAT",              C["GAT"],       "-"),
        ("gcn",           "GCN",              C["GCN"],       "--"),
        ("graphsage",     "GraphSAGE",        C["GraphSAGE"], ":"),
        ("gatv2",         "GATv2",            C["GATv2"],     "-."),
        ("gat_vanilla",   "GAT (vanilla)",    C["GAT_van"],   (0, (3, 1))),
    ]
    arch_data = {}
    for af, disp, col, ls in arch_specs:
        d = load_binned(af)
        if d is not None and len(d) > 0:
            arch_data[disp] = (d, col, ls)

    # Phase4 tabular + CNN binned
    p4b = TBL / "phase4_binned_metrics.csv"
    if p4b.exists():
        df4 = pd.read_csv(p4b)
        for model, col in [("XGBoost", C["XGBoost"]), ("Random Forest", C["RF"])]:
            sub = df4[df4["model"] == model]
            if len(sub) > 0:
                arch_data[model] = (sub, col, "-")
    cnnb = TBL / "phase4b_cnn_binned_metrics.csv"
    if cnnb.exists():
        arch_data["2D CNN"] = (pd.read_csv(cnnb), C["CNN"], (0, (5, 2)))

    if not arch_data:
        _skip("fig4a / fig4b", "no binned CSVs found"); return

    # ---- fig4a: MAE per bin (line plot) ----
    fig, ax = plt.subplots(figsize=(8, 6))
    for disp, (d, col, ls) in arch_data.items():
        ds = d.sort_values("bin") if "bin" in d.columns else d
        bins = ds["bin"].values if "bin" in ds else range(1, len(ds) + 1)
        ax.plot(bins, ds["mae"].values, marker="o", ms=5, lw=2,
                color=col, linestyle=ls, label=disp)
    ax.set_xlabel("Burn Probability Quintile Bin\n(1=lowest risk, 5=highest risk)")
    ax.set_ylabel("Mean Absolute Error (MAE)")
    ax.set_title("MAE by Risk Level\nHigh-risk tail (Bin 5) is most critical", fontweight="bold")
    ax.set_xticks([1, 2, 3, 4, 5])
    ax.legend(fontsize=8)
    plt.tight_layout(); save(fig, "fig4a_mae_by_bin.png")

    # ---- fig4b: bin-5 comparison (bar) ----
    models_b, vals_b, cols_b = [], [], []
    for disp, (d, col, ls) in arch_data.items():
        ds = d.sort_values("bin") if "bin" in d.columns else d
        b5 = ds[ds["bin"] == 5] if "bin" in ds.columns else ds.tail(1)
        if len(b5) > 0:
            models_b.append(disp); vals_b.append(float(b5["mae"].values[0])); cols_b.append(col)
    order = np.argsort(vals_b)
    models_b = [models_b[i] for i in order]
    vals_b   = [vals_b[i] for i in order]
    cols_b   = [cols_b[i] for i in order]

    fig, ax = plt.subplots(figsize=(8, 6))
    bars = ax.barh(models_b, vals_b, color=cols_b, height=0.66, alpha=0.9,
                   edgecolor="black", linewidth=0.5)
    for bar, disp in zip(bars, models_b):
        if disp == "GAT":
            bar.set_edgecolor("black"); bar.set_linewidth(2.2)
    for bar, v in zip(bars, vals_b):
        ax.text(v + 0.0005, bar.get_y() + bar.get_height() / 2, f"{v:.4f}",
                va="center", fontsize=9)
    ax.set_xlabel("MAE in Highest Risk Bin (BP > 0.047)")
    ax.set_title("High-Risk Bin Performance\nBin 5: BP ∈ [0.047, 0.208]", fontweight="bold")
    plt.tight_layout(); save(fig, "fig4b_highrisk_bin5.png")


# ════════════════════════════════════════════════════════════════════════════
# FIGURE 5 — Feature importance (RF + XGBoost -> 2 separate PNGs)
# ════════════════════════════════════════════════════════════════════════════

def fig5_feature_importance():
    print("\n[Fig 5] Feature importance — 2 separate PNGs")
    df = load_importances()
    if df is None:
        _skip("fig5a / fig5b", "phase4_feature_importances.csv not found"); return

    group_colors = {
        "FSP": "#1a6faf", "CFL": "#e07b39", "IGNITION": "#7b3f99",
        "STRUCT": "#b84040", "DEM": "#2ecc71", "FUEL": "#888888",
        "INTERACT": "#c0a020",
    }

    for model, color, fname in [
        ("Random Forest", C["RF"],      "fig5a_rf_importance.png"),
        ("XGBoost",       C["XGBoost"], "fig5b_xgboost_importance.png"),
    ]:
        sub = df[df["model"] == model].head(15).sort_values("importance")
        if len(sub) == 0:
            _skip(fname, f"no rows for {model}"); continue
        fig, ax = plt.subplots(figsize=(8, 7))
        bars = ax.barh(sub["feature"], sub["importance"], color=color,
                       alpha=0.85, height=0.7)
        for bar, feat in zip(bars, sub["feature"]):
            for key, col in group_colors.items():
                if key in feat.upper():
                    bar.set_color(col); break
        ax.set_xlabel("Feature Importance")
        ax.set_title(f"{LABEL.get(model, model)}\nTop 15 features", fontweight="bold")
        plt.tight_layout(); save(fig, fname)


# ════════════════════════════════════════════════════════════════════════════
# FIGURE 6 — Calibration (3 separate PNGs)
# ════════════════════════════════════════════════════════════════════════════

def load_cal(arch_file: str) -> pd.DataFrame | None:
    p = TBL / f"phase5b_{arch_file}_calibration.csv"
    return pd.read_csv(p) if p.exists() else None


def fig6_calibration():
    print("\n[Fig 6] Calibration — up to 3 separate PNGs")
    gat_cal = load_cal("gat")

    # ---- fig6a: GAT reliability diagram ----
    if gat_cal is not None and {"stage"}.issubset(gat_cal.columns):
        try:
            before = gat_cal[gat_cal["stage"] == "before"].iloc[0]
            after  = gat_cal[gat_cal["stage"] == "after"].iloc[0]
            targets = [0.50, 0.90, 0.95]
            b = [before["picp_50"], before["picp_90"], before["picp_95"]]
            a = [after["picp_50"],  after["picp_90"],  after["picp_95"]]
            fig, ax = plt.subplots(figsize=(6.5, 6))
            ax.plot([0, 1], [0, 1], "k--", lw=1.5, label="Perfect calibration", zorder=5)
            ax.scatter(targets, b, color=C["before"], s=80, zorder=6, label="Before scaling")
            ax.scatter(targets, a, color=C["after"], s=80, zorder=6, marker="s",
                       label=f"After scaling (T={after['T']:.3f})")
            ax.plot(targets, b, color=C["before"], lw=1.5, alpha=0.7)
            ax.plot(targets, a, color=C["after"], lw=1.5, alpha=0.7)
            ax.set_xlim(0.4, 1.02); ax.set_ylim(0.4, 1.02)
            ax.set_xlabel("Expected Coverage"); ax.set_ylabel("Actual Coverage (PICP)")
            ax.set_title("GAT Reliability Diagram\nTemperature Scaling", fontweight="bold")
            ax.legend(fontsize=9)
            plt.tight_layout(); save(fig, "fig6a_gat_reliability.png")
        except Exception as e:
            _skip("fig6a_gat_reliability.png", f"columns missing ({e})")
    else:
        _skip("fig6a_gat_reliability.png", "phase5b_gat_calibration.csv not found")

    # ---- fig6b: PICP-90 before/after all arches ----
    arch_labels, files = ["GAT", "GCN", "GraphSAGE"], ["gat", "gcn", "graphsage"]
    pb, pa, Ts = [], [], []
    for af in files:
        cal = load_cal(af)
        if cal is not None and "stage" in cal.columns:
            b = cal[cal["stage"] == "before"].iloc[0]; a = cal[cal["stage"] == "after"].iloc[0]
            pb.append(b["picp_90"]); pa.append(a["picp_90"]); Ts.append(a["T"])
        else:
            pb.append(np.nan); pa.append(np.nan); Ts.append(np.nan)
    if any(np.isfinite(pb)):
        x = np.arange(len(arch_labels)); w = 0.35
        fig, ax = plt.subplots(figsize=(7, 6))
        ax.bar(x - w/2, pb, w, color=C["before"], alpha=0.75, label="Before scaling", zorder=3)
        ax.bar(x + w/2, pa, w, color=C["after"], alpha=0.75, label="After scaling", zorder=3)
        ax.axhline(0.90, color="black", lw=1.5, ls="--", label="Target PICP=0.90", zorder=4)
        ax.fill_between([-0.5, len(arch_labels)-0.5], 0.85, 0.95, alpha=0.1, color="green")
        ax.set_xticks(x)
        ax.set_xticklabels([f"{l}\n(T={t:.3f})" if np.isfinite(t) else l
                            for l, t in zip(arch_labels, Ts)], fontsize=9)
        ax.set_ylabel("PICP at 90% Nominal Coverage")
        ax.set_title("PICP-90% Before/After\nAll architectures", fontweight="bold")
        ax.set_ylim(0.5, 1.05); ax.legend(fontsize=8)
        plt.tight_layout(); save(fig, "fig6b_picp90.png")
    else:
        _skip("fig6b_picp90.png", "no calibration CSVs found")

    # ---- fig6c: ECE comparison ----
    df = load_full_comparison()
    if not df.empty and "ece" in df.columns:
        d = _order_models(df)
        labels = [LABEL.get(m, m) for m in d["model"]]
        colors = [COLOR_FOR.get(m, "#aaaaaa") for m in d["model"]]
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.bar(labels, d["ece"].values, color=colors, alpha=0.85,
               edgecolor="black", linewidth=0.5)
        ax.axhline(0.05, color="red", lw=1.5, ls="--", label="ECE=0.05 threshold")
        ax.set_ylabel("ECE (Expected Calibration Error)")
        ax.set_title("ECE Comparison\nLower = better calibrated", fontweight="bold")
        ax.tick_params(axis="x", rotation=35)
        ax.legend(fontsize=9)
        plt.tight_layout(); save(fig, "fig6c_ece_comparison.png")
    else:
        _skip("fig6c_ece_comparison.png", "comparison table missing")


# ════════════════════════════════════════════════════════════════════════════
# FIGURE 7 — Uncertainty decomposition (3 separate PNGs) — needs gat .npz
# ════════════════════════════════════════════════════════════════════════════

def fig7_uncertainty():
    print("\n[Fig 7] Uncertainty decomposition — needs phase5a_gat_preds.npz")
    gat = load_npz(PRED / "phase5a_gat_preds.npz")
    if gat is None or not {"y_true_bp", "y_pred_bp", "total_unc"}.issubset(gat):
        _skip("fig7a / fig7b / fig7c", "phase5a_gat_preds.npz missing required keys")
        return
    y_true, y_pred = gat["y_true_bp"], gat["y_pred_bp"]
    total = gat["total_unc"]
    epi = gat.get("std_pred"); ale = gat.get("aleatoric")
    rng = np.random.default_rng(42)
    idx = rng.choice(len(y_true), min(20_000, len(y_true)), replace=False)

    # fig7a total uncertainty vs true BP
    fig, ax = plt.subplots(figsize=(6.5, 6))
    sc = ax.scatter(y_true[idx], total[idx], c=y_pred[idx], cmap="YlOrRd",
                    s=2, alpha=0.25, rasterized=True, vmin=0, vmax=0.15)
    plt.colorbar(sc, ax=ax, label="Predicted BP")
    ax.set_xlabel("True Burn Probability"); ax.set_ylabel("Total Uncertainty (σ)")
    ax.set_title("Total Uncertainty vs True BP\nColour = predicted BP", fontweight="bold")
    plt.tight_layout(); save(fig, "fig7a_total_uncertainty.png")

    # fig7b epistemic vs aleatoric
    if epi is not None and ale is not None:
        fig, ax = plt.subplots(figsize=(6.5, 6))
        ax.scatter(epi[idx], ale[idx], c=y_true[idx], cmap="YlOrRd",
                   s=2, alpha=0.2, rasterized=True, vmin=0, vmax=0.15)
        ax.set_xlabel("Epistemic Uncertainty (MC Dropout σ)")
        ax.set_ylabel("Aleatoric Uncertainty (√exp(log_var))")
        ax.set_title("Uncertainty Decomposition\nColour = true BP", fontweight="bold")
        plt.tight_layout(); save(fig, "fig7b_decomposition.png")
    else:
        _skip("fig7b_decomposition.png", "epistemic/aleatoric keys missing")

    # fig7c error vs uncertainty
    fig, ax = plt.subplots(figsize=(6.5, 6))
    err = np.abs(y_true - y_pred)
    ax.scatter(total[idx], err[idx], c=y_true[idx], cmap="YlOrRd",
               s=2, alpha=0.2, rasterized=True, vmin=0, vmax=0.15)
    lim = float(max(total[idx].max(), err[idx].max()))
    ax.plot([0, lim], [0, lim], "k--", lw=1.5, label="|error|=σ (perfect)")
    ax.set_xlabel("Total Uncertainty (σ)"); ax.set_ylabel("|Prediction Error|")
    ax.set_title("Error vs Uncertainty\nPerfect calibration: diagonal", fontweight="bold")
    ax.legend(fontsize=9)
    plt.tight_layout(); save(fig, "fig7c_error_vs_uncertainty.png")


# ════════════════════════════════════════════════════════════════════════════
# FIGURE 8 — Intervention analysis (map + hist per scenario) — needs .npz + graph
# ════════════════════════════════════════════════════════════════════════════

def fig8_intervention():
    print("\n[Fig 8] Intervention — needs effect .npz files + graph")
    gp = DATA / "graph_data_enriched.pt"
    if not HAS_TORCH or not gp.exists():
        _skip("fig8*", "graph_data_enriched.pt or torch unavailable"); return
    g = torch.load(gp, map_location="cpu", weights_only=False)
    pos = g.pos[g.test_mask].numpy()

    scenarios = {
        "fuel_reduction_30pct":       ("Fuel Reduction 30% (CFL × 0.70)", C["fuel"],      "fuel"),
        "firebreak":                  ("Firebreak Strip (CFL = 0 in strip)", C["firebreak"], "firebreak"),
        "ignition_suppression_50pct": ("Ignition Suppression 50% (Ign × 0.50)", C["ignition"], "ignition"),
    }
    rng = np.random.default_rng(42)
    idx = rng.choice(len(pos), min(25_000, len(pos)), replace=False)

    # gather deltas to fix a shared colour scale
    effects = {}
    for key in scenarios:
        for prefix in ["phase5d_v2_", "phase5d_"]:
            p = PRED / f"{prefix}{key}_gat_effects.npz"
            if p.exists():
                effects[key] = load_npz(p); break
    if not effects:
        _skip("fig8*", "no intervention effect .npz files found"); return

    all_deltas = np.concatenate([e["delta_bp"] for e in effects.values()])
    p2, p98 = np.percentile(all_deltas, [2, 98])
    vmax = min(max(abs(p2), abs(p98), 0.003), 0.05)

    for key, effect in effects.items():
        label, color, short = scenarios[key]
        delta = effect["delta_bp"]
        sig = effect.get("significant_mask")
        mean_d = float(delta.mean())
        pct_sig = float(sig.mean() * 100) if sig is not None else float("nan")
        pct_r = float((delta < 0).mean() * 100)

        # map
        fig, ax = plt.subplots(figsize=(7, 6.5))
        sc = ax.scatter(pos[idx, 1], -pos[idx, 0], c=delta[idx], cmap="RdBu_r",
                        vmin=-vmax, vmax=vmax, s=2, alpha=0.7, rasterized=True)
        plt.colorbar(sc, ax=ax, label="Δ BP", shrink=0.7, pad=0.02)
        title = f"{label}\nMean Δ={mean_d:+.4f}"
        if np.isfinite(pct_sig): title += f"   Sig: {pct_sig:.1f}%"
        ax.set_title(title, fontweight="bold")
        ax.set_xlabel("Column"); ax.set_ylabel("Row (N→S)")
        plt.tight_layout(); save(fig, f"fig8_{short}_map.png")

        # histogram
        fig, ax = plt.subplots(figsize=(7, 5.5))
        ax.hist(delta, bins=70, color=color, alpha=0.8, edgecolor="none")
        ax.axvline(0, color="black", lw=1.5, ls="--")
        ax.axvline(mean_d, color="red", lw=2, label=f"Mean={mean_d:+.4f}")
        if "delta_bp_lo_90" in effect and "delta_bp_hi_90" in effect:
            lo_med = float(np.percentile(effect["delta_bp_lo_90"], 50))
            hi_med = float(np.percentile(effect["delta_bp_hi_90"], 50))
            ax.axvspan(lo_med, hi_med, alpha=0.15, color=color)
        ax.set_xlabel("Δ Burn Probability"); ax.set_ylabel("Node count")
        ax.set_title(f"Distribution of Effects\n{pct_r:.1f}% nodes reduced", fontweight="bold")
        ax.legend(fontsize=8)
        plt.tight_layout(); save(fig, f"fig8_{short}_hist.png")


# ════════════════════════════════════════════════════════════════════════════
# FIGURE 9 — Architecture ablation (4 metrics -> 4 separate PNGs)
# ════════════════════════════════════════════════════════════════════════════

def fig9_ablation():
    print("\n[Fig 9] Architecture ablation — 4 separate metric PNGs")
    df = load_full_comparison()
    if df.empty:
        _skip("fig9*", "no comparison table found"); return

    metrics = [
        ("r2",       "R²",         False, "fig9a_r2.png"),
        ("mae",      "MAE",        True,  "fig9b_mae.png"),
        ("spearman", "Spearman ρ", False, "fig9c_spearman.png"),
        ("ece",      "ECE",        True,  "fig9d_ece.png"),
    ]
    for col, ylabel, ascending, fname in metrics:
        if col not in df.columns:
            _skip(fname, f"column '{col}' missing"); continue
        d = df.sort_values(col, ascending=ascending)
        labels = [LABEL.get(m, m) for m in d["model"]]
        colors = [COLOR_FOR.get(m, "#aaaaaa") for m in d["model"]]
        vals = d[col].values
        fig, ax = plt.subplots(figsize=(8, 6.5))
        bars = ax.barh(labels, vals, color=colors, alpha=0.9, height=0.66,
                       edgecolor="black", linewidth=0.5)
        for bar, m in zip(bars, d["model"]):
            if m == "GAT":
                bar.set_edgecolor("black"); bar.set_linewidth(2.2)
        span = float(np.abs(vals).max()) if len(vals) else 1.0
        for bar, v in zip(bars, vals):
            ax.text(v + (span * 0.02 if v >= 0 else -span * 0.02),
                    bar.get_y() + bar.get_height() / 2, f"{v:.4f}",
                    va="center", ha="left" if v >= 0 else "right", fontsize=8)
        note = "Lower is better" if ascending else "Higher is better"
        ax.set_xlabel(ylabel)
        ax.set_title(f"{ylabel} — {note}", fontweight="bold")
        ax.invert_yaxis()
        plt.tight_layout(); save(fig, fname)


# ════════════════════════════════════════════════════════════════════════════
# FIGURE 10 — Training dynamics (one PNG per architecture)
# ════════════════════════════════════════════════════════════════════════════

def fig10_training():
    print("\n[Fig 10] Training dynamics — one PNG per architecture")
    specs = [
        ("gat",         "GAT",           C["GAT"]),
        ("gcn",         "GCN",           C["GCN"]),
        ("graphsage",   "GraphSAGE",     C["GraphSAGE"]),
        ("gatv2",       "GATv2",         C["GATv2"]),
        ("gat_vanilla", "GAT (vanilla)", C["GAT_van"]),
    ]
    any_done = False
    for af, disp, color in specs:
        h = load_history(af)
        if h is None:
            _skip(f"fig10_{af}_training.png", f"history CSV for {af} not found"); continue
        fig, ax = plt.subplots(figsize=(7, 5.5))
        ax.plot(h["epoch"], h["train_loss"], lw=2, color=color, alpha=0.9, label="Train loss")
        ax.plot(h["epoch"], h["val_loss"], lw=2, color=color, alpha=0.9, ls="--", label="Val loss")
        best_i = h["val_loss"].idxmin()
        best_e = int(h["epoch"][best_i]); best_v = float(h["val_loss"].min())
        ax.axvline(best_e, color="red", lw=1.5, ls=":",
                   label=f"Best val epoch {best_e}\nloss={best_v:.4f}")
        ft = float(h["train_loss"].iloc[-1]); fv = float(h["val_loss"].iloc[-1])
        ax.set_xlabel("Epoch"); ax.set_ylabel("Loss (MSE)")
        ax.set_title(f"{disp} Training\ntrain={ft:.4f}  val={fv:.4f}  gap={fv-ft:.4f}",
                     fontweight="bold")
        ax.legend(fontsize=9)
        plt.tight_layout(); save(fig, f"fig10_{af}_training.png")
        any_done = True
    if not any_done:
        _skip("fig10*", "no history CSVs found")


# ════════════════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════════════════

def main():
    print("\n" + "=" * 66)
    print("  Paper Figure Generation — SEPARATED (one plot per PNG)")
    print(f"  Output: {FIG}")
    print("=" * 66)

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

    figs = sorted(FIG.glob("fig*.png"))
    total_mb = sum(f.stat().st_size for f in figs) / 1024 ** 2
    print("\n" + "=" * 66)
    print(f"  Done. {len(figs)} PNGs in {FIG.name}/  ({total_mb:.1f} MB)")
    print("=" * 66)
    for f in figs:
        print(f"    {f.name:<40} {f.stat().st_size // 1024:>5} KB")


if __name__ == "__main__":
    main()