"""
Uncertainty Calibration — Phase 5B of wildfire-uncertainty-gnn.

WHAT THIS PHASE ADDRESSES
--------------------------
Gap 2 — Calibration:
  A model is calibrated if its expressed confidence matches its actual
  accuracy. If the model says "I predict BP=0.10 ± 0.02", the true value
  should fall within that interval 95% of the time for a 2-sigma interval.

  Existing models (XGBoost, CNN) produce point predictions only.
  The GAT outputs mean + log_var (aleatoric) + MC Dropout std (epistemic).
  Phase 5B verifies this uncertainty is CALIBRATED and applies
  Temperature Scaling to correct any overconfidence.

WHY TEMPERATURE SCALING
------------------------
Temperature Scaling (Guo et al., 2017) is the simplest and most reliable
post-hoc calibration method. It introduces a single scalar parameter T:

    scaled_log_var = log_var + 2 * log(T)
    equivalently: scaled_var = T² * var

When T > 1: variance increases → model becomes less confident → better
  calibrated if model was overconfident.
When T < 1: variance decreases → model becomes more confident.
When T = 1: no change.

Temperature is learned by minimising NLL on the VALIDATION set only.
The model weights are NEVER changed. This is purely post-hoc.

KEY METRIC — PICP (Prediction Interval Coverage Probability)
-------------------------------------------------------------
For a 90% prediction interval:
  interval = [mean - 1.645 * sigma, mean + 1.645 * sigma]
  PICP = fraction of test nodes where true value falls in interval

A well-calibrated model: PICP ≈ 0.90 for 90% interval.
Overconfident model:     PICP < 0.90 (interval too narrow)
Underconfident model:    PICP > 0.90 (interval too wide)

RELIABILITY DIAGRAM
-------------------
Plots expected coverage (x-axis) vs actual coverage (y-axis).
Perfect calibration: diagonal line (y=x).
Overconfident: curve below diagonal.
Underconfident: curve above diagonal.

REFERENCES
----------
Guo, C., et al. (2017). On calibration of modern neural networks. ICML.
Lakshminarayanan, B., et al. (2017). Simple and scalable predictive
  uncertainty estimation using deep ensembles. NeurIPS.
"""

from __future__ import annotations
import numpy as np
from scipy import stats as sp_stats
from scipy.optimize import minimize_scalar


# ════════════════════════════════════════════════════════════════════════════
# Temperature Scaling
# ════════════════════════════════════════════════════════════════════════════

class TemperatureScaling:
    """
    Post-hoc temperature scaling calibration for GNN uncertainty.

    Learns a single scalar T that scales the predicted variance:
        calibrated_std = T * original_std

    T is fit on the VALIDATION set by minimising Gaussian NLL.
    Applied to TEST set predictions — no model retraining.

    Usage
    -----
    ts = TemperatureScaling()
    ts.fit(mean_pred_val, std_pred_val, y_true_val)
    calibrated_std_test = ts.scale(std_pred_test)
    """

    def __init__(self):
        self.T: float = 1.0          # learned temperature
        self.fitted: bool = False

    def _nll(self, T: float, mean: np.ndarray,
             std: np.ndarray, y: np.ndarray) -> float:
        """Gaussian NLL with scaled std. Minimise over T."""
        scaled_std = T * std + 1e-8
        nll = 0.5 * np.mean(
            np.log(2 * np.pi * scaled_std**2) +
            (y - mean)**2 / scaled_std**2
        )
        return float(nll)

    def fit(
        self,
        mean_pred: np.ndarray,
        std_pred:  np.ndarray,
        y_true:    np.ndarray,
    ) -> "TemperatureScaling":
        """
        Fit temperature on validation set predictions (transformed scale).

        Parameters
        ----------
        mean_pred : (N_val,) model mean predictions (transformed scale)
        std_pred  : (N_val,) MC Dropout std (epistemic, transformed scale)
        y_true    : (N_val,) true values (transformed scale)
        """
        result = minimize_scalar(
            self._nll,
            bounds   = (0.01, 10.0),
            method   = "bounded",
            args     = (mean_pred, std_pred, y_true),
        )
        self.T      = float(result.x)
        self.fitted = True
        print(f"  Temperature T = {self.T:.4f}")
        if self.T > 1.1:
            print(f"  → Model was overconfident. Intervals widened by {self.T:.2f}×")
        elif self.T < 0.9:
            print(f"  → Model was underconfident. Intervals narrowed by {1/self.T:.2f}×")
        else:
            print(f"  → Model was well-calibrated. T ≈ 1 (no major adjustment)")
        return self

    def scale(self, std_pred: np.ndarray) -> np.ndarray:
        """Apply temperature scaling to std predictions."""
        assert self.fitted, "Call fit() first"
        return self.T * std_pred

    def scale_total(
        self,
        epistemic: np.ndarray,
        aleatoric: np.ndarray,
    ) -> np.ndarray:
        """
        Scale total uncertainty = sqrt((T*ep)² + al²).
        Only epistemic is scaled — aleatoric comes from model log_var.
        """
        assert self.fitted, "Call fit() first"
        scaled_ep = self.T * epistemic
        return np.sqrt(scaled_ep**2 + aleatoric**2)


# ════════════════════════════════════════════════════════════════════════════
# Calibration Metrics
# ════════════════════════════════════════════════════════════════════════════

def compute_picp(
    y_true:    np.ndarray,
    mean_pred: np.ndarray,
    std_pred:  np.ndarray,
    coverage:  float = 0.90,
) -> float:
    """
    Prediction Interval Coverage Probability (PICP).

    PICP = fraction of test nodes where true value falls inside the
    predicted interval [mean ± z * std] where z = norm.ppf((1+cov)/2).

    A calibrated model: PICP ≈ coverage.
    Overconfident model: PICP < coverage (intervals too narrow).

    Parameters
    ----------
    y_true    : true values (original BP scale)
    mean_pred : predicted mean (original BP scale, inverse-transformed)
    std_pred  : predicted std (total uncertainty, original scale approx.)
    coverage  : target coverage (0.90 = 90% prediction interval)
    """
    z   = sp_stats.norm.ppf((1 + coverage) / 2)
    lo  = mean_pred - z * std_pred
    hi  = mean_pred + z * std_pred
    return float(np.mean((y_true >= lo) & (y_true <= hi)))


def compute_mpiw(
    mean_pred: np.ndarray,
    std_pred:  np.ndarray,
    coverage:  float = 0.90,
) -> float:
    """
    Mean Prediction Interval Width (MPIW).
    Narrower is better if PICP is satisfied.
    """
    z = sp_stats.norm.ppf((1 + coverage) / 2)
    return float(np.mean(2 * z * std_pred))


def reliability_curve(
    y_true:    np.ndarray,
    mean_pred: np.ndarray,
    std_pred:  np.ndarray,
    n_levels:  int = 19,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute reliability curve (calibration curve).

    For confidence levels [5%, 10%, ..., 95%], compute:
      expected_coverage = level
      actual_coverage   = PICP at that level

    Perfect calibration: actual = expected (diagonal).
    Overconfident: actual < expected (below diagonal).

    Returns
    -------
    expected : (n_levels,) array of target coverage values
    actual   : (n_levels,) array of measured PICP values
    """
    expected = np.linspace(0.05, 0.95, n_levels)
    actual   = np.array([
        compute_picp(y_true, mean_pred, std_pred, cov)
        for cov in expected
    ])
    return expected, actual


def compute_ace(
    expected: np.ndarray,
    actual:   np.ndarray,
) -> float:
    """
    Average Coverage Error (ACE) — area between reliability curve and diagonal.
    ACE = 0: perfectly calibrated.
    ACE > 0: systematically underconfident.
    ACE < 0: systematically overconfident.
    """
    return float(np.mean(actual - expected))


def compute_ence(
    y_true:    np.ndarray,
    mean_pred: np.ndarray,
    std_pred:  np.ndarray,
    n_bins:    int = 10,
) -> float:
    """
    Expected Normalised Calibration Error (ENCE) for regression.

    Bins predictions by predicted std, then measures whether the
    RMSE within each bin matches the mean predicted std:
        ENCE = (1/n_bins) Σ |RMSE_b - mean_std_b| / mean_std_b
    """
    order = np.argsort(std_pred)
    bins  = np.array_split(order, n_bins)
    ence  = 0.0
    for b in bins:
        if len(b) < 2:
            continue
        rmse_b    = float(np.sqrt(np.mean((y_true[b] - mean_pred[b])**2)))
        std_b     = float(np.mean(std_pred[b]))
        if std_b > 0:
            ence += abs(rmse_b - std_b) / std_b
    return ence / n_bins


def compute_all_calibration_metrics(
    y_true:    np.ndarray,
    mean_pred: np.ndarray,
    std_pred:  np.ndarray,
    label:     str = "model",
    verbose:   bool = True,
) -> dict:
    """
    Compute the full calibration metric suite.

    Metrics:
      PICP-50%  : 50% interval coverage
      PICP-90%  : 90% interval coverage (primary)
      PICP-95%  : 95% interval coverage
      MPIW-90%  : mean prediction interval width at 90%
      ACE       : average coverage error (reliability curve)
      ENCE      : expected normalised calibration error

    All metrics in original BP scale.
    """
    expected, actual = reliability_curve(y_true, mean_pred, std_pred)

    metrics = {
        "label":    label,
        "picp_50":  compute_picp(y_true, mean_pred, std_pred, 0.50),
        "picp_90":  compute_picp(y_true, mean_pred, std_pred, 0.90),
        "picp_95":  compute_picp(y_true, mean_pred, std_pred, 0.95),
        "mpiw_90":  compute_mpiw(mean_pred, std_pred, 0.90),
        "ace":      compute_ace(expected, actual),
        "ence":     compute_ence(y_true, mean_pred, std_pred),
        "expected": expected,
        "actual":   actual,
    }

    if verbose:
        print(f"\n  Calibration metrics — {label}")
        print(f"    PICP-50%  = {metrics['picp_50']:.4f}  "
              f"(target 0.500, {'✓' if abs(metrics['picp_50']-0.5)<0.05 else '✗'})")
        print(f"    PICP-90%  = {metrics['picp_90']:.4f}  "
              f"(target 0.900, {'✓' if abs(metrics['picp_90']-0.9)<0.05 else '✗'})")
        print(f"    PICP-95%  = {metrics['picp_95']:.4f}  "
              f"(target 0.950, {'✓' if abs(metrics['picp_95']-0.95)<0.05 else '✗'})")
        print(f"    MPIW-90%  = {metrics['mpiw_90']:.5f}  (interval width)")
        print(f"    ACE       = {metrics['ace']:+.4f}  "
              f"({'underconfident' if metrics['ace']>0 else 'overconfident' if metrics['ace']<-0.01 else 'calibrated'})")
        print(f"    ENCE      = {metrics['ence']:.4f}  (target < 0.10)")

    return metrics