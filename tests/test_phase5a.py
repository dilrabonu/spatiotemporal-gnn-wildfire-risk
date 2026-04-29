"""
Phase 5A Tests — run: pytest tests/test_phase5a.py -v

Groups:
  A — Model architecture (GAT, GCN, GraphSAGE)
  B — Gaussian NLL loss correctness
  C — MC Dropout (dropout ON at inference)
  D — Training pipeline
  E — Integration: saved checkpoint verification
"""

from __future__ import annotations
import sys
import pytest
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from wildfire_gnn.models.gnn import (
    build_model, gaussian_nll_loss, count_parameters,
    GATWildfire, GCNWildfire, GraphSAGEWildfire,
)


# ── Fixtures ──────────────────────────────────────────────────────────────

def make_mini_graph(n: int = 100, f: int = 61):
    """Minimal synthetic graph for unit tests."""
    from torch_geometric.data import Data
    x          = torch.randn(n, f)
    # 8-connected grid edges for a 10×10 grid
    rows, cols = [], []
    for i in range(n):
        for j in range(n):
            if i != j and abs(i//10 - j//10) <= 1 and abs(i%10 - j%10) <= 1:
                rows.append(i); cols.append(j)
    edge_index = torch.tensor([rows, cols], dtype=torch.long)
    y          = torch.randn(n, 1)
    y_raw      = torch.rand(n, 1) * 0.25

    train_mask = torch.zeros(n, dtype=torch.bool)
    val_mask   = torch.zeros(n, dtype=torch.bool)
    test_mask  = torch.zeros(n, dtype=torch.bool)
    train_mask[:70] = True
    val_mask[70:85] = True
    test_mask[85:]  = True

    return Data(x=x, edge_index=edge_index, y=y, y_raw=y_raw,
                train_mask=train_mask, val_mask=val_mask, test_mask=test_mask)


# ════════════════════════════════════════════════════════════════════════════
# GROUP A — Architecture correctness
# ════════════════════════════════════════════════════════════════════════════

class TestArchitectures:

    @pytest.mark.parametrize("arch", ["GAT", "GCN", "GraphSAGE"])
    def test_output_shape(self, arch):
        """All architectures must output (N,) mean and (N,) log_var."""
        model = build_model(arch, in_channels=61, hidden=64, num_layers=2, heads=4)
        data  = make_mini_graph()
        model.eval()
        with torch.no_grad():
            mean, log_var = model(data.x, data.edge_index)
        assert mean.shape == (100,),    f"{arch} mean shape wrong: {mean.shape}"
        assert log_var.shape == (100,), f"{arch} log_var shape wrong: {log_var.shape}"

    @pytest.mark.parametrize("arch", ["GAT", "GCN", "GraphSAGE"])
    def test_output_finite(self, arch):
        """Outputs must not contain NaN or Inf."""
        model = build_model(arch, in_channels=61, hidden=64, num_layers=2, heads=4)
        data  = make_mini_graph()
        model.eval()
        with torch.no_grad():
            mean, log_var = model(data.x, data.edge_index)
        assert torch.isfinite(mean).all(),    f"{arch} mean has NaN/Inf"
        assert torch.isfinite(log_var).all(), f"{arch} log_var has NaN/Inf"

    def test_gat_has_more_params_than_gcn(self):
        """GAT (with attention) should have more params than GCN."""
        gat = build_model("GAT", in_channels=61, hidden=64, heads=4, num_layers=2)
        gcn = build_model("GCN", in_channels=61, hidden=64, num_layers=2)
        assert count_parameters(gat) > count_parameters(gcn), \
            "GAT should have more parameters than GCN (attention weights)"

    def test_wrong_architecture_raises(self):
        with pytest.raises(ValueError, match="Unknown architecture"):
            build_model("LSTM", in_channels=61)

    def test_in_channels_mismatch_raises(self):
        """Model with wrong in_channels must fail on forward pass."""
        model = build_model("GCN", in_channels=32, hidden=64, num_layers=2)
        data  = make_mini_graph(n=50, f=61)  # 61 features, model expects 32
        model.eval()
        with pytest.raises(RuntimeError):
            model(data.x, data.edge_index)

    def test_log_var_clamped(self):
        """log_var must be clamped to [-10, 10] — numerical stability."""
        model = build_model("GAT", in_channels=61, hidden=64, heads=4, num_layers=2)
        # Extreme input to trigger clamping
        data  = make_mini_graph()
        data.x = data.x * 1000.0
        model.eval()
        with torch.no_grad():
            _, log_var = model(data.x, data.edge_index)
        assert log_var.min() >= -10.0 - 1e-5, "log_var min below -10"
        assert log_var.max() <= 10.0  + 1e-5, "log_var max above +10"


# ════════════════════════════════════════════════════════════════════════════
# GROUP B — Gaussian NLL loss
# ════════════════════════════════════════════════════════════════════════════

class TestGaussianNLLLoss:

    def test_loss_scalar(self):
        """Loss must return a scalar tensor."""
        mean    = torch.randn(100)
        log_var = torch.zeros(100)
        target  = torch.randn(100)
        loss    = gaussian_nll_loss(mean, log_var, target)
        assert loss.dim() == 0, "Loss must be scalar"
        assert torch.isfinite(loss), "Loss must be finite"

    def test_perfect_prediction_lower_loss(self):
        """Perfect prediction should give lower loss than random."""
        target   = torch.randn(200)
        mean_perfect = target.clone()
        mean_random  = torch.randn(200)
        log_var  = torch.zeros(200)
        loss_perfect = gaussian_nll_loss(mean_perfect, log_var, target).item()
        loss_random  = gaussian_nll_loss(mean_random,  log_var, target).item()
        assert loss_perfect < loss_random, \
            "Perfect prediction must have lower NLL than random"

    def test_nll_lower_than_mse_for_uncertain_data(self):
        """
        Gaussian NLL should be lower than MSE for data with high variance.
        NLL can adapt log_var; MSE cannot.
        """
        target   = torch.randn(200) * 5.0  # high variance
        mean     = torch.zeros(200)
        log_var_adaptive = torch.full((200,), 3.0)   # allows high variance
        log_var_fixed    = torch.zeros(200)           # assumes unit variance
        loss_adaptive = gaussian_nll_loss(mean, log_var_adaptive, target).item()
        loss_fixed    = gaussian_nll_loss(mean, log_var_fixed,    target).item()
        assert loss_adaptive < loss_fixed, \
            "Adaptive variance should give lower NLL for high-variance target"

    def test_mse_fallback_works(self):
        """MSE loss option must produce finite scalar."""
        mean   = torch.randn(50)
        target = torch.randn(50)
        loss   = nn.MSELoss()(mean, target)
        assert loss.dim() == 0
        assert torch.isfinite(loss)


# ════════════════════════════════════════════════════════════════════════════
# GROUP C — MC Dropout
# ════════════════════════════════════════════════════════════════════════════

class TestMCDropout:

    def test_dropout_on_gives_variance(self):
        """
        With dropout ON, repeated forward passes must produce different outputs.
        This is the core of MC Dropout — non-zero variance = epistemic uncertainty.
        """
        model = build_model("GAT", in_channels=61, hidden=64, heads=4,
                            num_layers=2, dropout=0.5)
        data  = make_mini_graph()
        model.train()  # dropout ON

        preds = []
        with torch.no_grad():
            for _ in range(10):
                mean, _ = model(data.x, data.edge_index)
                preds.append(mean.numpy())

        preds = np.stack(preds)  # (10, N)
        var_across_samples = preds.var(axis=0)
        assert var_across_samples.mean() > 0, \
            "MC Dropout: variance must be > 0 with dropout ON"

    def test_dropout_off_gives_zero_variance(self):
        """
        With dropout OFF (model.eval()), repeated passes must give identical output.
        """
        model = build_model("GAT", in_channels=61, hidden=64, heads=4,
                            num_layers=2, dropout=0.5)
        data  = make_mini_graph()
        model.eval()  # dropout OFF

        preds = []
        with torch.no_grad():
            for _ in range(5):
                mean, _ = model(data.x, data.edge_index)
                preds.append(mean.numpy())

        preds = np.stack(preds)
        max_diff = np.abs(preds - preds[0]).max()
        assert max_diff < 1e-6, \
            f"With dropout OFF, all passes must be identical (max_diff={max_diff})"

    def test_mc_samples_give_uncertainty_estimate(self):
        """
        MC Dropout std across samples must be > 0 on real forward passes.
        """
        model = build_model("GCN", in_channels=61, hidden=64, num_layers=2,
                            dropout=0.5)
        data  = make_mini_graph()
        model.train()

        samples = []
        with torch.no_grad():
            for _ in range(20):
                mean, _ = model(data.x, data.edge_index)
                samples.append(mean.numpy())

        samples = np.stack(samples)
        std     = samples.std(axis=0)
        assert std.mean() > 0, "MC Dropout std must be > 0"
        assert std.min() >= 0, "Std cannot be negative"


# ════════════════════════════════════════════════════════════════════════════
# GROUP D — Training pipeline
# ════════════════════════════════════════════════════════════════════════════

class TestTrainingPipeline:

    def _make_config(self, arch="GCN", epochs=3, loss="gaussian_nll"):
        return {
            "model": {
                "architecture": arch,
                "in_channels":  61,
                "hidden_channels": 64,
                "num_layers": 2,
                "heads": 4,
                "dropout": 0.3,
            },
            "training": {
                "epochs": epochs, "lr": 1e-3,
                "weight_decay": 1e-5, "patience": 5,
                "min_delta": 1e-4, "gradient_clip": 1.0,
            },
            "uncertainty": {"loss_function": loss},
        }

    def test_train_runs_without_error(self):
        """Full train loop must complete without exceptions."""
        from wildfire_gnn.models.gnn_pipeline import GNNPipeline
        config   = self._make_config(epochs=3)
        pipeline = GNNPipeline(config)
        data     = make_mini_graph()
        out      = pipeline.train(data)
        assert "history" in out
        assert "best_val_loss" in out
        assert out["best_val_loss"] < float("inf")

    def test_history_has_correct_length(self):
        """History length must equal epochs run (≤ max_epochs)."""
        from wildfire_gnn.models.gnn_pipeline import GNNPipeline
        config   = self._make_config(epochs=4, arch="GCN")
        pipeline = GNNPipeline(config)
        data     = make_mini_graph()
        out      = pipeline.train(data)
        assert len(out["history"]) <= 4

    def test_mse_loss_option(self):
        """Training with MSE loss must also run without error."""
        from wildfire_gnn.models.gnn_pipeline import GNNPipeline
        config   = self._make_config(epochs=2, loss="mse", arch="GCN")
        pipeline = GNNPipeline(config)
        data     = make_mini_graph()
        out      = pipeline.train(data)
        assert out["best_val_loss"] < float("inf")

    def test_early_stopping_works(self):
        """Early stopping must trigger before max_epochs if val_loss plateaus."""
        from wildfire_gnn.models.gnn_pipeline import EarlyStopping
        stopper = EarlyStopping(patience=3, min_delta=0.0)
        model   = build_model("GCN", in_channels=61, hidden=64, num_layers=1)
        stopped = False
        for i in range(20):
            if stopper.step(val_loss=1.0, model=model):  # constant loss
                stopped = True
                break
        assert stopped, "Early stopping must trigger on constant val_loss"

    def test_gradient_clip_applied(self):
        """Gradient norms after clipping must be ≤ clip_value."""
        model     = build_model("GCN", in_channels=61, hidden=64, num_layers=1)
        data      = make_mini_graph()
        optimizer = torch.optim.Adam(model.parameters(), lr=1.0)  # large lr
        model.train()
        mean, lv = model(data.x, data.edge_index)
        target   = data.y[data.train_mask].squeeze()
        loss     = gaussian_nll_loss(
            mean[data.train_mask], lv[data.train_mask], target
        )
        loss.backward()
        clip_val = 1.0
        nn.utils.clip_grad_norm_(model.parameters(), clip_val)
        for p in model.parameters():
            if p.grad is not None:
                assert p.grad.norm().item() <= clip_val + 1e-5, \
                    f"Gradient norm {p.grad.norm():.3f} exceeds clip {clip_val}"


# ════════════════════════════════════════════════════════════════════════════
# GROUP E — Saved checkpoint integration
# ════════════════════════════════════════════════════════════════════════════

class TestSavedCheckpoint:
    """
    Load saved GAT checkpoint and verify correctness.
    Skipped if checkpoint not yet generated.
    """
    CKPT = ROOT / "checkpoints" / "gnn_gat_best.pt"

    @pytest.fixture(autouse=True)
    def skip_if_no_ckpt(self):
        if not self.CKPT.exists():
            pytest.skip("gnn_gat_best.pt not yet generated — "
                        "run phase5a_train_gnn.py first")

    def test_checkpoint_loads(self):
        ckpt = torch.load(self.CKPT, map_location="cpu", weights_only=False)
        assert "model_state" in ckpt
        assert "config"      in ckpt

    def test_model_restores_from_checkpoint(self):
        ckpt  = torch.load(self.CKPT, map_location="cpu", weights_only=False)
        cfg   = ckpt["config"]["model"]
        model = build_model(
            cfg["architecture"], cfg["in_channels"],
            cfg["hidden_channels"],
            num_layers = cfg.get("num_layers", 4),
            heads      = cfg.get("heads", 8),
            dropout    = cfg.get("dropout", 0.3),
        )
        model.load_state_dict(ckpt["model_state"])
        # Forward pass must work after restore
        data = make_mini_graph()
        model.eval()
        with torch.no_grad():
            mean, lv = model(data.x, data.edge_index)
        assert torch.isfinite(mean).all()

    def test_saved_metrics_exist(self):
        metrics_path = ROOT / "reports" / "tables" / "phase5a_gat_metrics.csv"
        assert metrics_path.exists(), "phase5a_gat_metrics.csv not found"
        import pandas as pd
        df = pd.read_csv(metrics_path)
        assert "r2" in df.columns
        assert len(df) > 0

    def test_r2_above_floor(self):
        """
        After training, GAT R² must at least beat the linear baseline (Ridge R²=0.136).
        If this fails, the model did not learn anything from the graph.
        """
        import pandas as pd
        metrics_path = ROOT / "reports" / "tables" / "phase5a_gat_metrics.csv"
        if not metrics_path.exists():
            pytest.skip("Metrics CSV not found")
        df  = pd.read_csv(metrics_path)
        r2  = float(df["r2"].iloc[0])
        assert r2 > 0.136, \
            f"GAT R²={r2:.4f} does not beat linear baseline (Ridge=0.136). " \
            f"Check training: features, split, config."