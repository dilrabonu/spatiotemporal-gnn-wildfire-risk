"""
Phase 5D Tests — run: pytest tests/test_phase5d.py -v

Groups:
  A — Feature modification functions
  B — Intervention effect computation
  C — Summary statistics
  D — Integration: load effects NPZ and verify
"""
from __future__ import annotations
import sys
import numpy as np
import torch
import pytest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from wildfire_gnn.models.intervention import (
    get_feature_index,
    modify_feature_multiplicative,
    modify_feature_absolute,
    build_row_band_mask,
    build_region_mask,
    summarise_effect,
)


def make_graph_x(n: int = 200, f: int = 61, seed: int = 42):
    rng = np.random.default_rng(seed)
    return torch.tensor(rng.standard_normal((n, f)), dtype=torch.float32)


# ════════════════════════════════════════════════════════════════════════════
# GROUP A — Feature modification
# ════════════════════════════════════════════════════════════════════════════

class TestFeatureModification:

    def test_multiplicative_reduces_by_30pct(self):
        x   = make_graph_x(100, 61)
        idx = 0
        x_orig_mean = float(x[:, idx].mean())
        x_new = modify_feature_multiplicative(x, idx, delta_fraction=-0.30)
        x_new_mean = float(x_new[:, idx].mean())
        assert abs(x_new_mean / x_orig_mean - 0.70) < 0.01, \
            f"Expected 30% reduction, got {x_new_mean/x_orig_mean:.3f}"

    def test_multiplicative_does_not_modify_other_features(self):
        x   = make_graph_x(100, 61)
        idx = 5
        x_new = modify_feature_multiplicative(x, idx, delta_fraction=-0.30)
        for other_idx in [0, 1, 2, 3, 10, 20, 60]:
            if other_idx == idx:
                continue
            torch.testing.assert_close(x[:, other_idx], x_new[:, other_idx])

    def test_multiplicative_with_mask(self):
        x    = make_graph_x(100, 61)
        mask = torch.zeros(100, dtype=torch.bool)
        mask[:50] = True
        idx  = 0
        x_new = modify_feature_multiplicative(x, idx, -0.30, node_mask=mask)
        # Masked nodes should be reduced
        ratio_masked   = (x_new[mask, idx] / x[mask, idx]).mean().item()
        ratio_unmasked = (x_new[~mask, idx] / x[~mask, idx]).mean().item()
        assert abs(ratio_masked   - 0.70) < 0.01
        assert abs(ratio_unmasked - 1.00) < 1e-5

    def test_absolute_sets_exact_value(self):
        x   = make_graph_x(100, 61)
        idx = 2
        x_new = modify_feature_absolute(x, idx, new_value=0.0)
        assert float(x_new[:, idx].abs().max()) == pytest.approx(0.0, abs=1e-6)

    def test_absolute_with_mask_preserves_others(self):
        x    = make_graph_x(100, 61)
        mask = torch.zeros(100, dtype=torch.bool)
        mask[:30] = True
        idx  = 0
        x_new = modify_feature_absolute(x, idx, 0.0, node_mask=mask)
        assert float(x_new[mask, idx].abs().max()) == pytest.approx(0.0, abs=1e-6)
        torch.testing.assert_close(x[~mask, idx], x_new[~mask, idx])

    def test_multiplicative_returns_new_tensor(self):
        """Original graph.x must NOT be modified."""
        x     = make_graph_x(100, 61)
        x_ref = x.clone()
        _     = modify_feature_multiplicative(x, 0, -0.30)
        torch.testing.assert_close(x, x_ref)

    def test_absolute_returns_new_tensor(self):
        x     = make_graph_x(100, 61)
        x_ref = x.clone()
        _     = modify_feature_absolute(x, 0, 0.0)
        torch.testing.assert_close(x, x_ref)


class TestSpatialMasks:

    def make_pos(self, n: int = 200):
        rows = torch.arange(n) % 100
        cols = torch.arange(n) // 100
        return torch.stack([rows, cols], dim=1).float()

    def test_row_band_mask_coverage(self):
        pos  = self.make_pos(200)
        mask = build_row_band_mask(pos, row_min=20, row_max=40)
        assert mask.dtype == torch.bool
        rows_in = pos[mask, 0]
        assert float(rows_in.min()) >= 20
        assert float(rows_in.max()) <= 40

    def test_row_band_mask_excludes_outside(self):
        pos  = self.make_pos(200)
        mask = build_row_band_mask(pos, row_min=50, row_max=60)
        rows_out = pos[~mask, 0]
        assert not any((rows_out >= 50) & (rows_out <= 60))

    def test_region_mask_correct(self):
        pos  = self.make_pos(200)
        mask = build_region_mask(pos, 10, 30, 0, 1)
        inside = pos[mask]
        assert float(inside[:, 0].min()) >= 10
        assert float(inside[:, 0].max()) <= 30


class TestFeatureIndex:

    def test_finds_correct_index(self):
        names = ["CFL", "FSP_Index", "Ignition_Prob", "Struct_Exp_Index"]
        assert get_feature_index("CFL", names) == 0
        assert get_feature_index("Ignition_Prob", names) == 2

    def test_raises_on_missing(self):
        names = ["CFL", "FSP_Index"]
        with pytest.raises(ValueError, match="not found"):
            get_feature_index("NONEXISTENT", names)


# ════════════════════════════════════════════════════════════════════════════
# GROUP B — Intervention effect
# ════════════════════════════════════════════════════════════════════════════

class TestInterventionEffect:

    def make_effect(self, n=500, seed=42):
        rng   = np.random.default_rng(seed)
        y_orig = rng.exponential(0.024, n).clip(0, 0.25).astype(np.float32)
        delta  = rng.normal(-0.002, 0.005, n).astype(np.float32)
        y_new  = np.clip(y_orig + delta, 0, 1)
        lo     = delta - 0.01
        hi     = delta + 0.01
        sig    = (lo > 0) | (hi < 0)
        return {
            "delta_bp":         delta,
            "delta_std_bp":     np.abs(rng.normal(0.003, 0.001, n)),
            "delta_bp_lo_90":   lo,
            "delta_bp_hi_90":   hi,
            "y_orig_bp":        y_orig,
            "y_new_bp":         y_new,
            "significant_mask": sig,
            "delta_samples_bp": np.stack([delta + rng.normal(0, 0.001, n)
                                          for _ in range(10)]),
        }

    def test_summary_keys_present(self):
        effect  = self.make_effect()
        summary = summarise_effect(
            effect, "Test", 500, verbose=False
        )
        required = ["scenario", "n_test", "mean_delta_bp",
                    "pct_reduced", "pct_significant"]
        for k in required:
            assert k in summary, f"Missing key: {k}"

    def test_pct_reduced_valid(self):
        effect  = self.make_effect()
        summary = summarise_effect(effect, "Test", 500, verbose=False)
        assert 0 <= summary["pct_reduced"] <= 100

    def test_negative_delta_gives_high_pct_reduced(self):
        """If all deltas are negative → pct_reduced ≈ 100."""
        rng = np.random.default_rng(0)
        n   = 500
        y_orig = rng.exponential(0.024, n).astype(np.float32)
        delta  = -np.abs(rng.normal(0.005, 0.001, n)).astype(np.float32)
        effect = {
            "delta_bp":         delta,
            "delta_std_bp":     np.full(n, 0.001),
            "delta_bp_lo_90":   delta - 0.002,
            "delta_bp_hi_90":   delta + 0.002,
            "y_orig_bp":        y_orig,
            "y_new_bp":         y_orig + delta,
            "significant_mask": np.ones(n, dtype=bool),
            "delta_samples_bp": np.stack([delta]*5),
        }
        summary = summarise_effect(effect, "Test", n, verbose=False)
        assert summary["pct_reduced"] == pytest.approx(100.0, abs=0.1)


# ════════════════════════════════════════════════════════════════════════════
# GROUP D — Integration: saved NPZ
# ════════════════════════════════════════════════════════════════════════════

class TestIntegration:
    FUEL_NPZ = ROOT / "reports" / "predictions" / \
               "phase5d_fuel_reduction_30pct_gat_effects.npz"

    @pytest.fixture(autouse=True)
    def skip_if_missing(self):
        if not self.FUEL_NPZ.exists():
            pytest.skip(f"{self.FUEL_NPZ.name} not found — "
                        "run phase5d_intervention.py first")

    def test_npz_has_required_keys(self):
        d = np.load(self.FUEL_NPZ)
        for k in ["delta_bp", "delta_std_bp", "delta_bp_lo_90",
                  "delta_bp_hi_90", "y_orig_bp", "y_new_bp",
                  "significant_mask", "delta_samples_bp"]:
            assert k in d.files, f"Missing: {k}"

    def test_delta_shapes_consistent(self):
        d      = np.load(self.FUEL_NPZ)
        n_test = len(d["delta_bp"])
        assert d["y_orig_bp"].shape   == (n_test,)
        assert d["y_new_bp"].shape    == (n_test,)
        assert d["delta_std_bp"].shape == (n_test,)

    def test_fuel_reduction_mostly_decreases_bp(self):
        """30% fuel reduction should reduce BP for most nodes."""
        d       = np.load(self.FUEL_NPZ)
        delta   = d["delta_bp"]
        pct_neg = float((delta < 0).mean() * 100)
        assert pct_neg > 40, \
            f"Only {pct_neg:.1f}% of nodes show BP reduction — expected > 40%"

    def test_confidence_intervals_ordered(self):
        d = np.load(self.FUEL_NPZ)
        assert np.all(d["delta_bp_lo_90"] <= d["delta_bp_hi_90"]), \
            "Lower bound must be ≤ upper bound"

    def test_y_orig_in_valid_range(self):
        d = np.load(self.FUEL_NPZ)
        assert float(d["y_orig_bp"].min()) >= 0
        assert float(d["y_orig_bp"].max()) <= 1.0