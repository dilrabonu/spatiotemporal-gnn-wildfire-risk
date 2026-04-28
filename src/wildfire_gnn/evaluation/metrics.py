# src/wildfire_gnn/evaluation/metrics.py
"""
Evaluation Framework — Phase 4 of wildfire-uncertainty-gnn.

WHY THIS FRAMEWORK EXISTS
--------------------------
Every model in Phase 4 and Phase 5 is evaluated with IDENTICAL metrics.
This prevents cherry-picking and ensures fair comparison for the paper.

METRICS
-------
Primary (for model ranking):
  R²         — coefficient of determination (goodness of fit)
  MAE        — mean absolute error on original BP scale
  Spearman ρ — rank correlation (robust to outliers, good for skewed targets)

Calibration (for Gap 2):
  ECE        — Expected Calibration Error (requires uncertainty estimates)
  Brier Score — proper scoring rule for probabilistic predictions

Binned (for Gap 1 — high-risk tail matters most):
  R² and MAE per quintile of burn probability
  The highest quintile (BP > Q80) is the most important for management

CRITICAL RULE
-------------
ALWAYS apply inverse_transform before computing metrics.
Report metrics on the ORIGINAL burn probability scale, not the
quantile-transformed scale. R² on transformed scale ≠ R² on BP scale.
"""

from __future__ import annotations
from typing import Optional
import numpy as np
from scipy import stats


def inverse_transform_predictions(
    y_pred_t: np.ndarray,
    transformer_path: str,
) -> np.ndarray:
    """
    Apply inverse QuantileTransformer to model predictions.

    ALWAYS call this before computing any metric.
    Predictions from the model are in transformed space (near-Gaussian).
    Metrics must be reported in original burn probability space [0, ~0.25].

    Parameters
    ----------
    y_pred_t : predictions in transformed space, shape (N,) or (N,1)
    transformer_path : path to target_transformer.pkl

    Returns
    -------
    y_pred_bp : predictions in original BP scale, shape (N,)
    """
    import pickle
    from pathlib import Path
    with open(Path(transformer_path), "rb") as f:
        transformer = pickle.load(f)
    y_pred_t = np.asarray(y_pred_t).ravel()
    return transformer.inverse_transform(y_pred_t).ravel()


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Coefficient of determination R².

    R² = 1 - SS_res / SS_tot
    Range: (-∞, 1]. 1 = perfect. 0 = predicts mean. <0 = worse than mean.

    Under a strict geographic split, a naive model predicting the
    training mean for all test nodes gives R² ≈ -0.3 to -0.8 because
    the test BP distribution differs from training. Any positive R²
    under geographic split is a meaningful result.
    """
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    return float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0


def mae_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Error on original BP scale."""
    return float(np.mean(np.abs(
        np.asarray(y_true).ravel() - np.asarray(y_pred).ravel()
    )))


def spearman_rho(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Spearman rank correlation coefficient.

    Robust to outliers and distribution shape.
    Particularly appropriate for burn probability because:
    1. BP is right-skewed — rank correlation is more stable than Pearson
    2. Wildfire management cares about ranking cells by risk, not exact values
    3. A model that correctly identifies high-risk cells is useful even if
       the exact probability values are miscalibrated
    """
    rho, p_val = stats.spearmanr(
        np.asarray(y_true).ravel(),
        np.asarray(y_pred).ravel()
    )
    return float(rho)


def brier_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Brier Score — mean squared error for probabilistic predictions.

    BS = (1/N) Σ (p_pred - y_true)²
    Range: [0, 1]. Lower is better. 0 = perfect.

    For burn probability: y_true ∈ [0, 0.25], y_pred ∈ [0, 0.25].
    The Brier score rewards both accuracy AND calibration simultaneously.
    An overconfident model that always predicts 0.5 when truth is 0.024
    will have a very high Brier score.
    """
    y_true = np.asarray(y_true).ravel()
    y_pred = np.clip(np.asarray(y_pred).ravel(), 0.0, 1.0)
    return float(np.mean((y_pred - y_true) ** 2))


def expected_calibration_error(
    y_true:   np.ndarray,
    y_pred:   np.ndarray,
    n_bins:   int = 15,
) -> float:
    """
    Expected Calibration Error (ECE) adapted for regression.

    Standard ECE is for classification (confidence bins). We adapt for
    burn probability regression by binning predictions into n_bins equal-
    width bins and measuring the gap between predicted mean and actual
    mean within each bin.

    ECE = Σ_b (n_b / N) × |mean_pred_b - mean_true_b|

    Range: [0, 1]. Lower is better. ECE < 0.05 = well-calibrated.

    Note: This is a simplified calibration measure. For full probabilistic
    calibration evaluation in Phase 5, we use MC Dropout uncertainty
    estimates and the standard confidence-interval coverage metric.
    """
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    N      = len(y_true)

    # Bin by predicted value
    bins = np.linspace(y_pred.min(), y_pred.max() + 1e-10, n_bins + 1)
    ece  = 0.0

    for i in range(n_bins):
        mask = (y_pred >= bins[i]) & (y_pred < bins[i+1])
        n_b  = mask.sum()
        if n_b == 0:
            continue
        mean_pred = y_pred[mask].mean()
        mean_true = y_true[mask].mean()
        ece += (n_b / N) * abs(mean_pred - mean_true)

    return float(ece)


def binned_metrics(
    y_true:   np.ndarray,
    y_pred:   np.ndarray,
    n_bins:   int = 5,
) -> list[dict]:
    """
    Compute R² and MAE per quantile bin of TRUE burn probability.

    WHY BINNED EVALUATION:
    The average R² across all nodes hides how models perform on high-risk
    cells. A model that perfectly predicts low-BP cells (85% of the
    landscape) but fails on high-BP cells (the 5-15% that matter for
    management) will have a high average R² but be useless for applications.

    Binned evaluation exposes this failure mode. The highest bin (BP > Q80)
    is the most important for wildfire management decisions.

    Returns
    -------
    list of dicts, one per bin, with keys:
        bin_low, bin_high, n, r2, mae, spearman
    """
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()

    quantiles = np.linspace(0, 100, n_bins + 1)
    thresholds = np.percentile(y_true, quantiles)

    results = []
    for i in range(n_bins):
        lo = thresholds[i]
        hi = thresholds[i+1]
        # Include upper bound in last bin
        mask = (y_true >= lo) & (y_true <= hi if i == n_bins-1 else y_true < hi)
        n = mask.sum()
        if n < 5:
            continue
        results.append({
            "bin":      i + 1,
            "bin_low":  float(lo),
            "bin_high": float(hi),
            "n":        int(n),
            "r2":       r2_score(y_true[mask], y_pred[mask]),
            "mae":      mae_score(y_true[mask], y_pred[mask]),
            "spearman": spearman_rho(y_true[mask], y_pred[mask]),
        })

    return results


def compute_all_metrics(
    y_true:      np.ndarray,
    y_pred:      np.ndarray,
    model_name:  str = "model",
    verbose:     bool = True,
) -> dict:
    """
    Compute the full evaluation metric suite.

    Parameters
    ----------
    y_true     : true burn probability values (ORIGINAL scale, inverse-transformed)
    y_pred     : predicted burn probability values (ORIGINAL scale, inverse-transformed)
    model_name : label for printing
    verbose    : print results table

    Returns
    -------
    dict with keys: r2, mae, spearman, brier, ece, binned
    """
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()

    metrics = {
        "model":    model_name,
        "r2":       r2_score(y_true, y_pred),
        "mae":      mae_score(y_true, y_pred),
        "spearman": spearman_rho(y_true, y_pred),
        "brier":    brier_score(y_true, y_pred),
        "ece":      expected_calibration_error(y_true, y_pred),
        "n_test":   len(y_true),
        "binned":   binned_metrics(y_true, y_pred),
    }

    if verbose:
        print(f"\n  ── {model_name} ──")
        print(f"    R²       = {metrics['r2']:>8.4f}   "
              f"(>0 = beats naive mean predictor)")
        print(f"    MAE      = {metrics['mae']:>8.5f}  "
              f"(burn prob scale ~[0, 0.25])")
        print(f"    Spearman = {metrics['spearman']:>8.4f}   "
              f"(rank correlation, robust)")
        print(f"    Brier    = {metrics['brier']:>8.5f}  "
              f"(lower = better calibration)")
        print(f"    ECE      = {metrics['ece']:>8.5f}  "
              f"(target < 0.05 after calibration)")
        print(f"    n_test   = {metrics['n_test']:>8,}")

        if metrics["binned"]:
            print(f"\n    Binned by BP quantile:")
            print(f"    {'Bin':<6} {'BP range':<22} {'n':>8} "
                  f"{'R²':>8} {'MAE':>10} {'Spearman':>10}")
            print(f"    {'-'*68}")
            for b in metrics["binned"]:
                label = " ← HIGH RISK" if b["bin"] == len(metrics["binned"]) else ""
                print(f"    {b['bin']:<6} "
                      f"[{b['bin_low']:.4f}, {b['bin_high']:.4f}]  "
                      f"{b['n']:>8,} "
                      f"{b['r2']:>8.3f} "
                      f"{b['mae']:>10.5f} "
                      f"{b['spearman']:>10.3f}{label}")

    return metrics


def print_comparison_table(results: list[dict]) -> None:
    """
    Print a side-by-side comparison table for all models.
    Used at the end of Phase 4 to produce the paper Table 2.
    """
    print("\n" + "="*80)
    print("  PHASE 4 BASELINE COMPARISON TABLE (test split, original BP scale)")
    print("="*80)
    print(f"  {'Model':<22} {'R²':>8} {'MAE':>10} {'Spearman':>10} "
          f"{'Brier':>10} {'ECE':>8}")
    print(f"  {'-'*75}")

    # Sort by R² descending
    sorted_results = sorted(results, key=lambda x: x.get("r2", -999), reverse=True)
    for r in sorted_results:
        print(f"  {r['model']:<22} "
              f"{r.get('r2', float('nan')):>8.4f} "
              f"{r.get('mae', float('nan')):>10.5f} "
              f"{r.get('spearman', float('nan')):>10.4f} "
              f"{r.get('brier', float('nan')):>10.5f} "
              f"{r.get('ece', float('nan')):>8.5f}")

    print("="*80)
    best = sorted_results[0]
    print(f"\n  Best model: {best['model']}  (R²={best['r2']:.4f})")
    print(f"  This is the floor the GNN (Phase 5) must beat.\n")