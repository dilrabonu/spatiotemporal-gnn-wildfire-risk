"""
Phase 5B Tests — run: pytest tests/test_phase5b.py -v

Groups:
  A — Temperature scaling correctness
  B — Calibration metrics (PICP, MPIW, ACE, ENCE)
  C — Reliability curve
  D — Integration: load predictions and run calibration
"""
from __future__ import annotations
import sys
import numpy as np
import pytest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from wildfire_gnn.models.calibration import (
    TemperatureScaling,
    compute_picp,
    compute_mpiw,
    compute_ace,
    compute_ence,
    reliability_curve,
    compute_all_calibration_metrics,
)


def make_synthetic(n=5000, seed=42):
    rng = np.random.default_rng(seed)
    y_true    = rng.exponential(0.024, n).clip(0, 0.25).astype(np.float32)
    mean_pred = y_true + rng.normal(0, 0.008, n).astype(np.float32)
    std_pred  = np.full(n, 0.012, dtype=np.float32)
    return y_true, mean_pred, std_pred


class TestTemperatureScaling:

    def test_fit_returns_positive_T(self):
        rng = np.random.default_rng(0)
        n   = 2000
        y   = rng.normal(0, 1, n).astype(np.float32)
        m   = y + rng.normal(0, 0.1, n).astype(np.float32)
        s   = np.full(n, 0.5, np.float32)  # std too small → T > 1
        ts  = TemperatureScaling()
        ts.fit(m, s, y)
        assert ts.T > 0, "Temperature must be positive"
        assert ts.fitted is True

    def test_overconfident_gives_T_greater_than_1(self):
        """Narrow std (overconfident) → T > 1."""
        rng = np.random.default_rng(1)
        n   = 3000
        y   = rng.normal(0, 1, n)
        m   = y + rng.normal(0, 0.1, n)
        s   = np.full(n, 0.1)   # very narrow intervals
        ts  = TemperatureScaling()
        ts.fit(m, s, y)
        assert ts.T > 1.0, f"Expected T>1 for overconfident model, got T={ts.T:.3f}"

    def test_underconfident_gives_T_less_than_1(self):
        """Very wide std (underconfident) → T < 1."""
        rng = np.random.default_rng(2)
        n   = 3000
        y   = rng.normal(0, 1, n)
        m   = y + rng.normal(0, 0.1, n)
        s   = np.full(n, 10.0)  # very wide intervals
        ts  = TemperatureScaling()
        ts.fit(m, s, y)
        assert ts.T < 1.0, f"Expected T<1 for underconfident model, got T={ts.T:.3f}"

    def test_well_calibrated_gives_T_near_1(self):
        """Perfectly calibrated → T ≈ 1."""
        rng = np.random.default_rng(3)
        n   = 5000
        y   = rng.normal(0, 1, n)
        m   = y + rng.normal(0, 0.1, n)
        s   = np.full(n, 1.0)   # std matches actual spread
        ts  = TemperatureScaling()
        ts.fit(m, s, y)
        assert 0.5 < ts.T < 2.0, f"T={ts.T:.3f} far from 1 for calibrated model"

    def test_scale_multiplies_by_T(self):
        ts = TemperatureScaling()
        ts.T      = 2.5
        ts.fitted = True
        std = np.array([0.1, 0.2, 0.3])
        scaled = ts.scale(std)
        np.testing.assert_allclose(scaled, std * 2.5, rtol=1e-5)

    def test_scale_requires_fit(self):
        ts = TemperatureScaling()
        with pytest.raises(AssertionError):
            ts.scale(np.array([0.1]))


class TestCalibrationMetrics:

    def test_picp_perfect_coverage(self):
        """Infinite intervals should give PICP=1.0."""
        n  = 1000
        yt = np.random.default_rng(0).normal(0, 1, n)
        m  = np.zeros(n)
        s  = np.full(n, 1000.0)   # huge intervals
        assert compute_picp(yt, m, s, 0.90) == pytest.approx(1.0, abs=0.01)

    def test_picp_zero_coverage(self):
        """Zero-width intervals give PICP ≈ 0."""
        n  = 1000
        yt = np.random.default_rng(0).normal(0, 1, n)
        m  = np.full(n, 999.0)    # predictions way off
        s  = np.full(n, 0.001)    # tiny intervals
        assert compute_picp(yt, m, s, 0.90) < 0.05

    def test_picp_range_valid(self):
        y, m, s = make_synthetic()
        picp = compute_picp(y, m, s, 0.90)
        assert 0.0 <= picp <= 1.0

    def test_mpiw_positive(self):
        y, m, s = make_synthetic()
        mpiw = compute_mpiw(m, s, 0.90)
        assert mpiw > 0

    def test_mpiw_scales_with_std(self):
        y, m, s = make_synthetic()
        mpiw_small = compute_mpiw(m, s * 0.5, 0.90)
        mpiw_large = compute_mpiw(m, s * 2.0, 0.90)
        assert mpiw_large > mpiw_small

    def test_ace_calibrated_model(self):
        """ACE near 0 for a calibrated model."""
        rng = np.random.default_rng(42)
        n   = 5000
        y   = rng.normal(0, 1, n)
        m   = rng.normal(0, 0.5, n)
        s   = np.full(n, 1.0)
        exp, act = reliability_curve(y, m, s, n_levels=9)
        ace = compute_ace(exp, act)
        assert abs(ace) < 0.5, f"ACE={ace:.3f} too large for near-calibrated model"

    def test_ence_nonnegative(self):
        y, m, s = make_synthetic()
        ence = compute_ence(y, m, s, n_bins=5)
        assert ence >= 0.0


class TestReliabilityCurve:

    def test_curve_length(self):
        y, m, s = make_synthetic()
        exp, act = reliability_curve(y, m, s, n_levels=9)
        assert len(exp) == 9
        assert len(act) == 9

    def test_expected_monotone(self):
        y, m, s = make_synthetic()
        exp, _ = reliability_curve(y, m, s, n_levels=9)
        assert np.all(np.diff(exp) > 0), "Expected coverage must be monotone"

    def test_actual_in_0_1(self):
        y, m, s = make_synthetic()
        _, act = reliability_curve(y, m, s, n_levels=9)
        assert np.all(act >= 0) and np.all(act <= 1)


class TestIntegration:
    """Load actual Phase 5A predictions and run calibration."""

    PRED_PATH = ROOT / "reports" / "predictions" / "phase5a_gat_preds.npz"

    @pytest.fixture(autouse=True)
    def skip_if_no_preds(self):
        if not self.PRED_PATH.exists():
            pytest.skip("phase5a_gat_preds.npz not found — "
                        "run phase5a_save_predictions.py first")

    def test_preds_npz_has_required_keys(self):
        data = np.load(self.PRED_PATH)
        required = ["y_true_bp", "y_pred_bp", "mean_pred_t",
                    "std_pred", "aleatoric", "total_unc",
                    "log_var_mean", "samples", "test_idx"]
        for k in required:
            assert k in data.files, f"Missing key in NPZ: {k}"

    def test_preds_shapes_consistent(self):
        data    = np.load(self.PRED_PATH)
        n_test  = len(data["y_true_bp"])
        n_mc    = data["samples"].shape[0]
        assert data["y_pred_bp"].shape  == (n_test,)
        assert data["std_pred"].shape   == (n_test,)
        assert data["samples"].shape    == (n_mc, n_test)
        assert n_mc == 30, f"Expected 30 MC passes, got {n_mc}"

    def test_r2_from_preds_is_correct(self):
        """R² computed from saved predictions must match confirmed result."""
        data     = np.load(self.PRED_PATH)
        y_true   = data["y_true_bp"]
        y_pred   = data["y_pred_bp"]
        ss_res   = np.sum((y_true - y_pred)**2)
        ss_tot   = np.sum((y_true - y_true.mean())**2)
        r2       = float(1 - ss_res / ss_tot)
        # GAT confirmed R²=0.7659 — allow small MC randomness
        assert r2 > 0.70, f"R²={r2:.4f} too low — wrong checkpoint loaded"

    def test_calibration_runs_on_real_preds(self):
        """Temperature scaling must fit and produce T in reasonable range."""
        data        = np.load(self.PRED_PATH)
        mean_pred_t = data["mean_pred_t"]
        std_pred    = data["std_pred"]
        aleatoric   = data["aleatoric"]
        total_std   = np.sqrt(std_pred**2 + aleatoric**2)
        # Use a subset as synthetic "val" to test fit
        n  = len(mean_pred_t)
        n_val = n // 5
        ts = TemperatureScaling()
        # Create synthetic y_true_t from samples mean
        y_t = data["samples"].mean(axis=0)
        ts.fit(mean_pred_t[:n_val], total_std[:n_val], y_t[:n_val])
        assert 0.01 <= ts.T <= 10.0, f"T={ts.T:.3f} out of range"