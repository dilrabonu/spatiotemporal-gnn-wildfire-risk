"""
Target variable engineering — Phase 2 of wildfire-uncertainty-gnn.

WHAT THIS FILE DOES
-------------------
Manages the QuantileTransformer for the burn probability (BP) target variable.

WHY TRANSFORMATION IS NEEDED
------------------------------
Raw BP is severely right-skewed (confirmed in Phase 2 dataset audit):
  min    = 0.0000036
  max    = 0.2509
  mean   = 0.0242
  median = 0.0120
  std    = 0.0328

Problems from this skew if you train without transformation:
  1. MSE loss dominated by a tiny fraction of high-BP cells.
  2. Gaussian NLL loss (our Gap 1 fix) assumes near-symmetric noise.
  3. Gradient norms across the [0.000003, 0.25] range are numerically unstable.

Solution: QuantileTransformer(output_distribution='normal') maps BP
to near-Gaussian (Q-Q r² = 0.998). After transform: mean ≈ 0, std ≈ 1.

CRITICAL RULE — NEVER DOUBLE-TRANSFORM
---------------------------------------
data.y in the PyG graph is ALREADY quantile-transformed after Phase 3.
DO NOT re-apply transform() at inference or evaluation time.
ONLY call inverse_transform() when you need to report metrics in
the original burn probability scale.

The transformer.pkl file is saved specifically for this inverse operation.

FAILURE FROM PREVIOUS PROJECT
------------------------------
A previous notebook cell re-applied QuantileTransformer during evaluation,
effectively double-transforming predictions. This made all reported metrics
meaningless (R² appeared 0.0, MAE appeared huge). Guard against this:
  assert abs(float(data.y.mean())) < 0.5, "y not transformed or double-transformed"
  assert 0.5 < float(data.y.std()) < 2.0, "y std out of expected range"
"""

from __future__ import annotations
import pickle
import warnings
from pathlib import Path
from typing import Optional
import numpy as np
from sklearn.preprocessing import QuantileTransformer

# Statistics from Phase 2 audit (for documentation and assertions)
BURN_PROB_MIN    = 3.64e-06
BURN_PROB_MAX    = 0.2509
BURN_PROB_MEAN   = 0.02417
BURN_PROB_STD    = 0.03283
BURN_PROB_MEDIAN = 0.01200


class TargetTransformer:
    """
    QuantileTransformer wrapper for burn probability target.

    Enforces correct usage: fit on train only, inverse at eval.
    """

    def __init__(self, n_quantiles: int = 1000, random_state: int = 42):
        self.n_quantiles  = n_quantiles
        self.random_state = random_state
        self._qt: Optional[QuantileTransformer] = None
        self._fitted = False

    # ── Fit ──────────────────────────────────────────────────────────────
    def fit(self, y: np.ndarray) -> "TargetTransformer":
        """
        Fit on TRAIN SPLIT burn probability values only.

        NEVER fit on val or test data — this causes target leakage.
        In Phase 3, this is called after the geographic split is created,
        using only the train-split BP values.

        Parameters
        ----------
        y : 1D array of burn probability values from training nodes
        """
        y = np.asarray(y, dtype=np.float64).ravel()
        n = len(y)
        if n < 100:
            warnings.warn(f"Fitting on only {n} samples — likely an error.")

        self._qt = QuantileTransformer(
            n_quantiles         = min(self.n_quantiles, n),
            output_distribution = "normal",
            random_state        = self.random_state,
        )
        self._qt.fit(y.reshape(-1, 1))
        self._fitted = True
        return self

    # ── Transform ────────────────────────────────────────────────────────
    def transform(self, y: np.ndarray) -> np.ndarray:
        """
        Map burn probability → near-Gaussian (applied once before training).

        After transformation: values are near standard normal (mean≈0, std≈1).
        """
        self._require_fitted()
        shape = y.shape
        return self._qt.transform(
            np.asarray(y, dtype=np.float64).reshape(-1, 1)
        ).reshape(shape)

    def fit_transform(self, y: np.ndarray) -> np.ndarray:
        """Convenience: fit then transform (TRAIN SPLIT ONLY)."""
        return self.fit(y).transform(y)

    # ── Inverse ──────────────────────────────────────────────────────────
    def inverse_transform(self, y_t: np.ndarray) -> np.ndarray:
        """
        Map model predictions back to original burn probability scale.

        ALWAYS call this before reporting R², MAE, Spearman ρ.
        Output will be in approximately [0.000003, 0.25].

        Parameters
        ----------
        y_t : 1D or 2D array of model outputs (transformed scale)
        """
        self._require_fitted()
        shape = y_t.shape
        return self._qt.inverse_transform(
            np.asarray(y_t, dtype=np.float64).reshape(-1, 1)
        ).reshape(shape)

    # ── Validation ───────────────────────────────────────────────────────
    def validate(self, y_transformed: np.ndarray) -> None:
        """
        Assert the transformed values look correct.

        Raises AssertionError if transform was applied twice or not at all.
        """
        y = np.asarray(y_transformed).ravel()
        assert np.all(np.isfinite(y)), \
            "Transformed target has NaN/Inf — check for corrupt data"
        mean = float(np.mean(y))
        std  = float(np.std(y))
        assert abs(mean) < 0.5, (
            f"Transformed mean={mean:.4f}, expected near 0. "
            "Was transform applied twice?"
        )
        assert 0.5 < std < 2.0, (
            f"Transformed std={std:.4f}, expected near 1. "
            "Was transform applied at all?"
        )
        print(f"  ✓  Target transform validated: mean={mean:.4f}, std={std:.4f}")

    # ── Persistence ──────────────────────────────────────────────────────
    def save(self, path: str | Path) -> None:
        """Save fitted transformer to .pkl file for inverse transform at eval time."""
        self._require_fitted()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)
        print(f"  ✓  Transformer saved: {path}")

    @classmethod
    def load(cls, path: str | Path) -> "TargetTransformer":
        """Load a previously saved transformer."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Transformer not found: {path}")
        with open(path, "rb") as f:
            obj = pickle.load(f)
        print(f"  ✓  Transformer loaded: {path}")
        return obj

    def _require_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError("Call fit() before transform/inverse_transform.")

    def __repr__(self) -> str:
        status = "fitted" if self._fitted else "not fitted"
        return f"TargetTransformer(n_quantiles={self.n_quantiles}, {status})"


def analyze_target_distribution(y: np.ndarray) -> dict:
    """
    Compute full distribution statistics for raw BP values.
    Prints summary and returns dict for report.
    """
    y = np.asarray(y, dtype=np.float64).ravel()
    stats = {
        "n":              len(y),
        "min":            float(np.min(y)),
        "max":            float(np.max(y)),
        "mean":           float(np.mean(y)),
        "median":         float(np.median(y)),
        "std":            float(np.std(y)),
        "pct_below_0.01": float(np.mean(y < 0.01) * 100),
        "pct_below_0.05": float(np.mean(y < 0.05) * 100),
        "pct_above_0.10": float(np.mean(y >= 0.10) * 100),
        "q10": float(np.percentile(y, 10)),
        "q25": float(np.percentile(y, 25)),
        "q75": float(np.percentile(y, 75)),
        "q90": float(np.percentile(y, 90)),
        "q95": float(np.percentile(y, 95)),
        "q99": float(np.percentile(y, 99)),
        "skewness": float(_skewness(y)),
    }
    print("\n  Burn Probability Distribution (raw):")
    for k, v in stats.items():
        if isinstance(v, float):
            print(f"    {k:<22} : {v:.6f}")
        else:
            print(f"    {k:<22} : {v:,}")
    return stats


def _skewness(x: np.ndarray) -> float:
    mu, sigma = np.mean(x), np.std(x)
    return float(np.mean(((x - mu) / sigma) ** 3)) if sigma > 0 else 0.0