"""
GNN Training Pipeline — Phase 5A.

Handles: training loop, validation, early stopping,
MC Dropout inference, metric computation.

CRITICAL RULES
--------------
1. Dropout stays ON during MC Dropout inference (model.train())
2. NEVER re-apply QuantileTransformer at eval — y is already transformed
3. ALWAYS inverse_transform before reporting metrics
4. Early stopping on val_loss — patience=20
5. Gradient clip at 1.0 to prevent exploding gradients
"""

from __future__ import annotations
import time
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.data import Data

from wildfire_gnn.models.gnn import gaussian_nll_loss, count_parameters
from wildfire_gnn.evaluation.metrics import (
    r2_score, mae_score, spearman_rho, brier_score,
    expected_calibration_error, binned_metrics,
)


class EarlyStopping:
    """Stop training when val_loss stops improving."""
    def __init__(self, patience: int = 20, min_delta: float = 1e-4):
        self.patience   = patience
        self.min_delta  = min_delta
        self.best_loss  = float("inf")
        self.counter    = 0
        self.best_state: Optional[dict] = None

    def step(self, val_loss: float, model: nn.Module) -> bool:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss  = val_loss
            self.counter    = 0
            self.best_state = {k: v.cpu().clone()
                               for k, v in model.state_dict().items()}
            return False   # continue
        self.counter += 1
        return self.counter >= self.patience  # True = stop

    def restore_best(self, model: nn.Module) -> None:
        if self.best_state is not None:
            model.load_state_dict(self.best_state)


class GNNPipeline:
    """
    Training and evaluation pipeline for all GNN architectures.

    Usage
    -----
    pipeline = GNNPipeline(config)
    outputs  = pipeline.train(data, stage="stage1")
    metrics  = pipeline.evaluate(data)
    """

    def __init__(self, config: dict):
        self.config = config
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.history: dict[str, list] = {
            "epoch": [], "train_loss": [], "val_loss": []
        }
        self.model: Optional[nn.Module] = None

    # ── Build ──────────────────────────────────────────────────────────────
    def build_model(self) -> nn.Module:
        from wildfire_gnn.models.gnn import build_model
        m_cfg = self.config["model"]
        model = build_model(
            architecture = m_cfg["architecture"],
            in_channels  = m_cfg["in_channels"],
            hidden       = m_cfg["hidden_channels"],
            num_layers   = m_cfg.get("num_layers", 4),
            heads        = m_cfg.get("heads", 8),
            dropout      = m_cfg.get("dropout", 0.3),
        )
        model = model.to(self.device)
        self.model = model
        n_params = count_parameters(model)
        print(f"  Model      : {model.name}")
        print(f"  Parameters : {n_params:,}")
        print(f"  Device     : {self.device}")
        return model

    # ── Train ──────────────────────────────────────────────────────────────
    def train(self, data: Data, stage: str = "stage1") -> dict:
        t_cfg = self.config["training"]
        if self.model is None:
            self.build_model()

        model      = self.model
        data       = data.to(self.device)
        epochs     = t_cfg.get("epochs", 200)
        lr         = t_cfg.get("lr", 1e-3)
        wd         = t_cfg.get("weight_decay", 1e-5)
        patience   = t_cfg.get("patience", 20)
        grad_clip  = t_cfg.get("gradient_clip", 1.0)
        loss_fn    = self.config["uncertainty"].get("loss_function", "gaussian_nll")

        optimizer  = torch.optim.Adam(
            model.parameters(), lr=lr, weight_decay=wd
        )
        scheduler  = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs
        )
        stopper    = EarlyStopping(
            patience  = patience,
            min_delta = t_cfg.get("min_delta", 1e-4),
        )

        y_train = data.y[data.train_mask].squeeze()
        y_val   = data.y[data.val_mask].squeeze()

        print(f"\n  Training ({epochs} epochs, patience={patience}, "
              f"loss={loss_fn}, device={self.device})")
        print(f"  {'Epoch':>6}  {'Train Loss':>12}  {'Val Loss':>10}")
        print(f"  {'-'*35}")

        t0 = time.time()
        for epoch in range(1, epochs + 1):
            # ── Train step ────────────────────────────────────────────────
            model.train()
            optimizer.zero_grad()
            mean, log_var = model(data.x, data.edge_index)

            if loss_fn == "gaussian_nll":
                loss = gaussian_nll_loss(
                    mean[data.train_mask],
                    log_var[data.train_mask],
                    y_train,
                )
            else:
                loss = nn.MSELoss()(mean[data.train_mask], y_train)

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            scheduler.step()

            # ── Val step ──────────────────────────────────────────────────
            model.eval()
            with torch.no_grad():
                mean_v, lv_v = model(data.x, data.edge_index)
            if loss_fn == "gaussian_nll":
                val_loss = gaussian_nll_loss(
                    mean_v[data.val_mask],
                    lv_v[data.val_mask],
                    y_val,
                ).item()
            else:
                val_loss = nn.MSELoss()(
                    mean_v[data.val_mask], y_val
                ).item()

            self.history["epoch"].append(epoch)
            self.history["train_loss"].append(loss.item())
            self.history["val_loss"].append(val_loss)

            if epoch % 10 == 0 or epoch == 1:
                print(f"  {epoch:>6}  {loss.item():>12.4f}  "
                      f"{val_loss:>10.4f}")

            if stopper.step(val_loss, model):
                print(f"\n  Early stopping at epoch {epoch}  "
                      f"(best val_loss={stopper.best_loss:.4f})")
                break

        stopper.restore_best(model)
        elapsed = time.time() - t0
        print(f"\n  ✓  Training complete: {elapsed/60:.1f} min  "
              f"best_val_loss={stopper.best_loss:.4f}")

        return {
            "history":       pd.DataFrame(self.history),
            "best_val_loss": stopper.best_loss,
            "epochs_run":    epoch,
        }

    # ── MC Dropout inference ──────────────────────────────────────────────
    def mc_dropout_predict(
        self,
        data:       Data,
        mask:       Tensor,
        n_samples:  int = 30,
    ) -> dict[str, np.ndarray]:
        """
        Run N forward passes with dropout ACTIVE to get epistemic uncertainty.

        CRITICAL: model.train() keeps dropout ON even during inference.
        This is the MC Dropout technique (Gal & Ghahramani, 2016).

        Returns
        -------
        dict with:
          mean_pred   : (N_mask,) mean of MC samples = final prediction
          std_pred    : (N_mask,) std of MC samples  = epistemic uncertainty
          aleatoric   : (N_mask,) sqrt(exp(log_var)) = aleatoric uncertainty
          total_unc   : (N_mask,) sqrt(aleatoric² + epistemic²)
          samples     : (n_samples, N_mask) all MC sample predictions
        """
        assert self.model is not None, "Call train() first"
        data = data.to(self.device)

        # Keep dropout ON for MC Dropout
        self.model.train()

        sample_means   = []
        sample_logvars = []

        with torch.no_grad():
            for _ in range(n_samples):
                mean, log_var = self.model(data.x, data.edge_index)
                sample_means.append(mean[mask].cpu().numpy())
                sample_logvars.append(log_var[mask].cpu().numpy())

        sample_means   = np.stack(sample_means)    # (n_samples, N_mask)
        sample_logvars = np.stack(sample_logvars)  # (n_samples, N_mask)

        mean_pred  = sample_means.mean(axis=0)
        std_pred   = sample_means.std(axis=0)       # epistemic
        aleatoric  = np.sqrt(np.exp(sample_logvars.mean(axis=0)))  # aleatoric
        total_unc  = np.sqrt(aleatoric**2 + std_pred**2)

        return {
            "mean_pred":  mean_pred,
            "std_pred":   std_pred,
            "aleatoric":  aleatoric,
            "total_unc":  total_unc,
            "samples":    sample_means,
        }

    # ── Evaluate ──────────────────────────────────────────────────────────
    def evaluate(
        self,
        data:             Data,
        transformer_path: str,
        n_mc_samples:     int = 30,
        verbose:          bool = True,
    ) -> dict:
        """
        Full evaluation on test split:
          1. MC Dropout predictions (mean + uncertainty)
          2. Inverse-transform to original BP scale
          3. Compute R², MAE, Spearman, Brier, ECE, binned

        CRITICAL: inverse_transform BEFORE any metric computation.
        """
        assert self.model is not None, "Call train() first"

        # MC Dropout predictions
        mc = self.mc_dropout_predict(data, data.test_mask, n_mc_samples)

        # Inverse-transform predictions
        with open(transformer_path, "rb") as f:
            transformer = pickle.load(f)

        y_pred_bp = transformer.inverse_transform(
            mc["mean_pred"].reshape(-1, 1)
        ).ravel()
        y_true_bp = data.y_raw[data.test_mask].cpu().numpy().ravel()

        if verbose:
            print(f"\n  Evaluation on test split "
                  f"(n={int(data.test_mask.sum()):,}, "
                  f"MC samples={n_mc_samples})")
            print(f"  y_true : mean={y_true_bp.mean():.5f}  "
                  f"max={y_true_bp.max():.4f}")
            print(f"  y_pred : mean={y_pred_bp.mean():.5f}  "
                  f"max={y_pred_bp.max():.4f}")

        metrics = {
            "model":     getattr(self.model, "name", "GNN"),
            "r2":        r2_score(y_true_bp, y_pred_bp),
            "mae":       mae_score(y_true_bp, y_pred_bp),
            "spearman":  spearman_rho(y_true_bp, y_pred_bp),
            "brier":     brier_score(y_true_bp, y_pred_bp),
            "ece":       expected_calibration_error(y_true_bp, y_pred_bp),
            "n_test":    int(data.test_mask.sum()),
            "binned":    binned_metrics(y_true_bp, y_pred_bp),
            "mc":        mc,
            "y_true_bp": y_true_bp,
            "y_pred_bp": y_pred_bp,
        }

        if verbose:
            print(f"\n  ── {metrics['model']} (test split) ──")
            print(f"    R²       = {metrics['r2']:>8.4f}")
            print(f"    MAE      = {metrics['mae']:>8.5f}")
            print(f"    Spearman = {metrics['spearman']:>8.4f}")
            print(f"    Brier    = {metrics['brier']:>8.5f}")
            print(f"    ECE      = {metrics['ece']:>8.5f}")
            beats_cnn = metrics['r2'] > 0.7187
            beats_xgb = metrics['r2'] > 0.6761
            print(f"\n    vs CNN baseline (R²=0.7187)  : "
                  f"{'✓ BEATS CNN' if beats_cnn else '✗ below CNN'}")
            print(f"    vs XGBoost baseline (R²=0.6761): "
                  f"{'✓ BEATS XGBoost' if beats_xgb else '✗ below XGBoost'}")

        return metrics

    # ── Save / Load ───────────────────────────────────────────────────────
    def save(self, path: str) -> None:
        assert self.model is not None
        torch.save({
            "model_state": self.model.state_dict(),
            "model_name":  self.model.name,
            "config":      self.config,
            "history":     self.history,
        }, path)
        print(f"  ✓  Model saved: {path}")

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        if self.model is None:
            self.build_model()
        self.model.load_state_dict(ckpt["model_state"])
        self.history = ckpt.get("history", {})
        print(f"  ✓  Model loaded: {path}")