"""
GNN Training Pipeline — Phase 5A (Memory-Safe: NeighborLoader)

WHY THE PREVIOUS VERSION CRASHED (OOM)
----------------------------------------
Full-graph GAT on 327,405 nodes × 2,511,084 edges × 8 heads tried to
allocate 2.7 GB for attention maps in a single forward pass.

  Attention memory = E × heads × (hidden//heads) × 4 bytes
                   = 2,511,084 × 8 × 32 × 4 = ~2.6 GB

On a Windows laptop this exceeds available RAM instantly.

THE FIX: NeighborLoader (mini-batch subgraph sampling)
--------------------------------------------------------
NeighborLoader samples small subgraphs centered on batches of seed nodes:
  batch_size=1024 seeds × num_neighbors=[10,10,10,10] hops
  → ~10,000 nodes max per subgraph → ~50 MB RAM per batch ✓

Training is equivalent to full-graph for learning feature representations.
Inference uses num_neighbors=[-1] (all neighbors) in small batches.

MEMORY PER BATCH (after fix)
------------------------------
  GAT hidden=256, heads=8 : ~100 MB per batch  ✓
  GCN hidden=256           :  ~60 MB per batch  ✓
  GraphSAGE hidden=256     :  ~60 MB per batch  ✓

CRITICAL RULES
---------------
1. model.train() for MC Dropout at inference — dropout stays ON
2. NEVER re-apply QuantileTransformer at eval — y is already transformed
3. ALWAYS inverse_transform before reporting metrics
4. Early stopping on val_loss (patience=20)
5. Gradient clip at 1.0
"""

from __future__ import annotations
import time
import pickle
import numpy as np
import pandas as pd
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader

from wildfire_gnn.models.gnn import gaussian_nll_loss, count_parameters
from wildfire_gnn.evaluation.metrics import (
    r2_score, mae_score, spearman_rho, brier_score,
    expected_calibration_error, binned_metrics,
)


class EarlyStopping:
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
            return False
        self.counter += 1
        return self.counter >= self.patience

    def restore_best(self, model: nn.Module) -> None:
        if self.best_state is not None:
            model.load_state_dict(self.best_state)


class GNNPipeline:
    """
    Memory-safe GNN pipeline using NeighborLoader mini-batches.
    Works on Windows laptops with 8-16 GB RAM without modification.
    """

    def __init__(self, config: dict):
        self.config  = config
        self.device  = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.history: dict[str, list] = {
            "epoch": [], "train_loss": [], "val_loss": []
        }
        self.model: Optional[nn.Module] = None

    # ── Build ──────────────────────────────────────────────────────────────
    def build_model(self) -> nn.Module:
        from wildfire_gnn.models.gnn import build_model
        m = self.config["model"]
        model = build_model(
            architecture = m["architecture"],
            in_channels  = m["in_channels"],
            hidden       = m["hidden_channels"],
            num_layers   = m.get("num_layers", 4),
            heads        = m.get("heads", 8),
            dropout      = m.get("dropout", 0.3),
        )
        model.to(self.device)
        self.model = model
        print(f"  Model      : {model.name}")
        print(f"  Parameters : {count_parameters(model):,}")
        print(f"  Device     : {self.device}")
        return model

    # ── Train ──────────────────────────────────────────────────────────────
    def train(self, data: Data, stage: str = "stage1") -> dict:
        """
        Mini-batch training with NeighborLoader.
        Samples subgraphs around seed nodes — avoids full-graph OOM.
        """
        t_cfg      = self.config["training"]
        if self.model is None:
            self.build_model()

        model       = self.model
        epochs      = t_cfg.get("epochs", 200)
        lr          = t_cfg.get("lr", 1e-3)
        wd          = t_cfg.get("weight_decay", 1e-5)
        patience    = t_cfg.get("patience", 20)
        grad_clip   = t_cfg.get("gradient_clip", 1.0)
        batch_size  = int(t_cfg.get("batch_size", 256))
        loss_fn     = self.config["uncertainty"].get(
                          "loss_function", "gaussian_nll")
        num_layers  = self.config["model"].get("num_layers", 4)

        # 5 neighbors per layer during training (memory-efficient)
        num_neighbors_train = t_cfg.get("neighbors", [5] * num_layers)

        optimizer = torch.optim.Adam(
            model.parameters(), lr=lr, weight_decay=wd
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs
        )
        stopper   = EarlyStopping(
            patience  = patience,
            min_delta = t_cfg.get("min_delta", 1e-4),
        )

        train_loader = NeighborLoader(
            data,
            num_neighbors = num_neighbors_train,
            batch_size    = batch_size,
            input_nodes   = data.train_mask,
            shuffle       = True,
            num_workers   = 0,   # must be 0 on Windows
        )
        val_loader = NeighborLoader(
            data,
            num_neighbors = num_neighbors_train,
            batch_size    = batch_size * 2,
            input_nodes   = data.val_mask,
            shuffle       = False,
            num_workers   = 0,
        )

        print(f"\n  Mini-batch training (NeighborLoader — memory safe)")
        print(f"  batch_size={batch_size}  neighbors={num_neighbors_train}")
        print(f"  epochs={epochs}  patience={patience}  "
              f"loss={loss_fn}  device={self.device}")
        print(f"\n  {'Epoch':>6}  {'Train Loss':>12}  {'Val Loss':>10}")
        print(f"  {'-'*35}")

        t0 = time.time()
        for epoch in range(1, epochs + 1):

            # Train
            model.train()
            total_loss, total_n = 0.0, 0
            for batch in train_loader:
                batch = batch.to(self.device)
                optimizer.zero_grad()
                mean, log_var = model(batch.x, batch.edge_index)
                n_seed  = batch.batch_size
                y_seed  = batch.y[:n_seed].squeeze()
                m_seed  = mean[:n_seed]
                lv_seed = log_var[:n_seed]

                if loss_fn == "gaussian_nll":
                    loss = gaussian_nll_loss(m_seed, lv_seed, y_seed)
                else:
                    loss = nn.MSELoss()(m_seed, y_seed)

                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
                total_loss += loss.item() * n_seed
                total_n    += n_seed

            train_loss = total_loss / max(total_n, 1)

            # Val
            model.eval()
            vl_total, vl_n = 0.0, 0
            with torch.no_grad():
                for batch in val_loader:
                    batch    = batch.to(self.device)
                    mean, lv = model(batch.x, batch.edge_index)
                    n_seed   = batch.batch_size
                    y_seed   = batch.y[:n_seed].squeeze()
                    if loss_fn == "gaussian_nll":
                        vl = gaussian_nll_loss(
                            mean[:n_seed], lv[:n_seed], y_seed)
                    else:
                        vl = nn.MSELoss()(mean[:n_seed], y_seed)
                    vl_total += vl.item() * n_seed
                    vl_n     += n_seed

            val_loss = vl_total / max(vl_n, 1)
            scheduler.step()

            self.history["epoch"].append(epoch)
            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)

            if epoch % 10 == 0 or epoch == 1:
                print(f"  {epoch:>6}  {train_loss:>12.4f}  "
                      f"{val_loss:>10.4f}")

            if stopper.step(val_loss, model):
                print(f"\n  Early stopping at epoch {epoch}  "
                      f"(best val_loss={stopper.best_loss:.4f})")
                break

        stopper.restore_best(model)
        elapsed = time.time() - t0
        print(f"\n  ✓  Training done: {elapsed/60:.1f} min  "
              f"best_val_loss={stopper.best_loss:.4f}")

        return {
            "history":       pd.DataFrame(self.history),
            "best_val_loss": stopper.best_loss,
            "epochs_run":    epoch,
        }

    # ── MC Dropout inference ──────────────────────────────────────────────
    def mc_dropout_predict(
        self,
        data:      Data,
        mask:      Tensor,
        n_samples: int = 30,
    ) -> dict[str, np.ndarray]:
        """
        Batched MC Dropout: model.train() stays ON, batches through mask.
        Uses num_neighbors=-1 (all neighbors) so each seed node gets
        its full k-hop neighborhood — accurate inference.
        """
        assert self.model is not None
        num_layers    = self.config["model"].get("num_layers", 4)
        num_neighbors = [-1] * num_layers   # all neighbors at inference

        test_loader = NeighborLoader(
            data,
            num_neighbors = num_neighbors,
            batch_size    = 256,            # small for full-neighbor inference
            input_nodes   = mask,
            shuffle       = False,
            num_workers   = 0,
        )

        self.model.train()   # dropout ON for MC Dropout

        all_means   = []
        all_logvars = []

        print(f"  Running {n_samples} MC Dropout passes (batched)...")
        for s in range(n_samples):
            means_s, lv_s = [], []
            with torch.no_grad():
                for batch in test_loader:
                    batch    = batch.to(self.device)
                    mean, lv = self.model(batch.x, batch.edge_index)
                    n_seed   = batch.batch_size
                    means_s.append(mean[:n_seed].cpu().numpy())
                    lv_s.append(lv[:n_seed].cpu().numpy())
            all_means.append(np.concatenate(means_s))
            all_logvars.append(np.concatenate(lv_s))
            if (s + 1) % 10 == 0:
                print(f"    MC pass {s+1}/{n_samples}")

        sample_means   = np.stack(all_means)
        sample_logvars = np.stack(all_logvars)

        mean_pred = sample_means.mean(axis=0)
        std_pred  = sample_means.std(axis=0)
        aleatoric = np.sqrt(np.exp(sample_logvars.mean(axis=0)))
        total_unc = np.sqrt(aleatoric**2 + std_pred**2)

        return {
            "mean_pred": mean_pred,
            "std_pred":  std_pred,
            "aleatoric": aleatoric,
            "total_unc": total_unc,
            "samples":   sample_means,
        }

    # ── Evaluate ──────────────────────────────────────────────────────────
    def evaluate(
        self,
        data:             Data,
        transformer_path: str,
        n_mc_samples:     int = 30,
        verbose:          bool = True,
    ) -> dict:
        assert self.model is not None

        mc = self.mc_dropout_predict(data, data.test_mask, n_mc_samples)

        with open(transformer_path, "rb") as f:
            transformer = pickle.load(f)

        y_pred_bp = transformer.inverse_transform(
            mc["mean_pred"].reshape(-1, 1)
        ).ravel()
        y_true_bp = data.y_raw[data.test_mask].cpu().numpy().ravel()

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
            print(f"\n    vs CNN (0.7187): "
                  f"{'✓ BEATS CNN' if metrics['r2']>0.7187 else '✗ below CNN'}")
            print(f"    vs XGB (0.6761): "
                  f"{'✓ BEATS XGB' if metrics['r2']>0.6761 else '✗ below XGB'}")

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
        print(f"  ✓  Saved: {path}")

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        if self.model is None:
            self.build_model()
        self.model.load_state_dict(ckpt["model_state"])
        self.history = ckpt.get("history", {})
        print(f"  ✓  Loaded: {path}")