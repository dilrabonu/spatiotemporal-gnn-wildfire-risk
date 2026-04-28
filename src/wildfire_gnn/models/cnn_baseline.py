"""
2D CNN Spatial Baseline — Phase 4.

WHY CNN IS A CRITICAL BASELINE (not just RF/XGBoost)
------------------------------------------------------
The CNN is the KEY architectural comparison for the paper.

RF and XGBoost: no spatial context at all
CNN: sees spatial context via convolution but NO graph topology
GNN: sees spatial context via message-passing WITH graph topology

The comparison chain:
  RF/XGBoost (no spatial) → CNN (spatial, no graph) → GNN (spatial + graph)

If GNN beats CNN: the GRAPH TOPOLOGY matters, not just spatial context.
If CNN beats GNN: the convolution extracts spatial patterns better than
  message-passing on this dataset. This would be a surprising and
  publishable finding in itself.

CNN DESIGN
----------
Input: patch of raster features centered on each node
  - We extract a 7×7 spatial patch around each subsampled node
  - Patch contains the 4 base rasters (not all 61 — those are node features)
  - This isolates the spatial convolution contribution
  - Same geographic split as all other models

Architecture: lightweight 2D CNN
  Conv(4→32, k=3) → BN → ReLU
  Conv(32→64, k=3) → BN → ReLU → MaxPool(2)
  Conv(64→128, k=3) → BN → ReLU
  GlobalAvgPool → FC(128→64) → ReLU → FC(64→1)

Why lightweight: training on 237,304 patches per epoch is expensive.
Keep CNN small enough to train in < 30 minutes on CPU.
GPU recommended but not required.

PATCH EXTRACTION
----------------
For each subsampled node at (row, col) in original raster space:
  - Extract 7×7 window from each aligned raster
  - Stack into (4, 7, 7) tensor: 4 channels = 4 base rasters
  - Pad with zeros at boundaries

The patch radius (3 pixels at stride=6) = 18m × 3 = 54m context.
This is the local neighborhood the CNN "sees" for each prediction.
"""

from __future__ import annotations
import time
import numpy as np
from pathlib import Path
from typing import Optional


# ── CNN Architecture ──────────────────────────────────────────────────────
def build_cnn_model(in_channels: int = 4, patch_size: int = 7) -> "torch.nn.Module":
    """
    Build lightweight 2D CNN for burn probability prediction from patches.

    Parameters
    ----------
    in_channels : number of raster channels (4 base rasters)
    patch_size  : spatial size of input patch (7 = 7×7 pixels)

    Returns
    -------
    torch.nn.Module
    """
    import torch.nn as nn

    class WildfireCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Sequential(
                nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(),
            )
            self.conv2 = nn.Sequential(
                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.MaxPool2d(2),
            )
            self.conv3 = nn.Sequential(
                nn.Conv2d(64, 128, kernel_size=3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(),
            )
            self.global_pool = nn.AdaptiveAvgPool2d(1)
            self.head = nn.Sequential(
                nn.Flatten(),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(64, 1),
            )

        def forward(self, x):
            x = self.conv1(x)
            x = self.conv2(x)
            x = self.conv3(x)
            x = self.global_pool(x)
            return self.head(x).squeeze(-1)

    return WildfireCNN()


# ── Patch extraction ──────────────────────────────────────────────────────
def extract_patches(
    rows_idx:    np.ndarray,
    cols_idx:    np.ndarray,
    aligned_dir: Path,
    patch_radius: int = 3,
    raster_names: list[str] = None,
) -> np.ndarray:
    """
    Extract (N, C, H, W) patch tensor for CNN input.

    For each node at (row, col), extracts a (2*patch_radius+1) × (2*patch_radius+1)
    spatial window from each base raster. Result shape: (N, 4, 7, 7) with default params.

    Parameters
    ----------
    rows_idx, cols_idx : node positions in original raster space
    aligned_dir        : path to aligned .tif files
    patch_radius       : half-size of patch (3 → 7×7 patch)
    raster_names       : which rasters to use (default: 4 base rasters)

    Returns
    -------
    patches : np.ndarray shape (N, C, patch_size, patch_size), float32
    """
    import rasterio

    if raster_names is None:
        raster_names = ["CFL", "FSP_Index", "Ignition_Prob", "Struct_Exp_Index"]

    P  = 2 * patch_radius + 1
    N  = len(rows_idx)
    C  = len(raster_names)

    # Load all rasters into memory
    raster_arrays = []
    for name in raster_names:
        path = aligned_dir / f"{name}.tif"
        with rasterio.open(path) as src:
            arr    = src.read(1).astype(np.float32)
            nodata = src.nodata
        if nodata is not None:
            arr[arr == nodata] = 0.0
        arr[arr < -1e30] = 0.0

        # Normalize per-raster to [0,1]
        vmin, vmax = arr.min(), arr.max()
        if vmax > vmin:
            arr = (arr - vmin) / (vmax - vmin)
        raster_arrays.append(arr)

    H, W = raster_arrays[0].shape

    # Pad each raster for boundary nodes
    padded = [np.pad(arr, patch_radius, mode='reflect') for arr in raster_arrays]

    # Extract patches
    patches = np.zeros((N, C, P, P), dtype=np.float32)
    for i, (r, c) in enumerate(zip(rows_idx.tolist(), cols_idx.tolist())):
        r_pad = r + patch_radius
        c_pad = c + patch_radius
        for ch, arr in enumerate(padded):
            patches[i, ch] = arr[r_pad-patch_radius:r_pad+patch_radius+1,
                                  c_pad-patch_radius:c_pad+patch_radius+1]

    print(f"  Patches extracted: {patches.shape}  "
          f"(N={N:,}, C={C}, patch={P}×{P})")
    return patches


# ── CNN Baseline Trainer ──────────────────────────────────────────────────
class CNNBaseline:
    """
    2D CNN spatial baseline trainer and predictor.

    Trains on patches extracted from aligned rasters.
    Uses the same geographic split as all other models.
    """

    def __init__(
        self,
        patch_radius:   int   = 3,
        epochs:         int   = 50,
        batch_size:     int   = 512,
        lr:             float = 1e-3,
        patience:       int   = 10,
        random_state:   int   = 42,
    ):
        self.patch_radius   = patch_radius
        self.epochs         = epochs
        self.batch_size     = batch_size
        self.lr             = lr
        self.patience       = patience
        self.random_state   = random_state
        self.name           = "2D CNN (spatial)"
        self.model          = None
        self.history_       = {"train_loss": [], "val_loss": []}

    def fit(
        self,
        X_train_patches: np.ndarray,
        y_train:         np.ndarray,
        X_val_patches:   np.ndarray,
        y_val:           np.ndarray,
        device_str:      str = "cpu",
    ) -> "CNNBaseline":
        """
        Train CNN on patch tensors.

        Parameters
        ----------
        X_train_patches : (N_train, C, H, W) float32 patch tensor
        y_train         : (N_train,) transformed targets
        X_val_patches   : (N_val, C, H, W) float32 patch tensor
        y_val           : (N_val,) transformed targets
        device_str      : "cuda" or "cpu"
        """
        import torch
        import torch.nn as nn
        from torch.utils.data import TensorDataset, DataLoader

        torch.manual_seed(self.random_state)
        device = torch.device(device_str)

        in_channels = X_train_patches.shape[1]
        self.model  = build_cnn_model(
            in_channels=in_channels,
            patch_size=2*self.patch_radius+1
        ).to(device)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.epochs
        )
        criterion = nn.MSELoss()

        # Datasets
        train_ds = TensorDataset(
            torch.tensor(X_train_patches, dtype=torch.float32),
            torch.tensor(y_train.ravel(), dtype=torch.float32),
        )
        val_ds = TensorDataset(
            torch.tensor(X_val_patches, dtype=torch.float32),
            torch.tensor(y_val.ravel(), dtype=torch.float32),
        )
        train_dl = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True)
        val_dl   = DataLoader(val_ds,   batch_size=self.batch_size, shuffle=False)

        best_val_loss  = float("inf")
        patience_count = 0
        t0             = time.time()

        for epoch in range(self.epochs):
            # Train
            self.model.train()
            train_loss = 0.0
            for xb, yb in train_dl:
                xb, yb = xb.to(device), yb.to(device)
                pred = self.model(xb)
                loss = criterion(pred, yb)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * len(xb)
            train_loss /= len(train_ds)

            # Validate
            self.model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for xb, yb in val_dl:
                    xb, yb = xb.to(device), yb.to(device)
                    pred = self.model(xb)
                    val_loss += criterion(pred, yb).item() * len(xb)
            val_loss /= len(val_ds)

            self.history_["train_loss"].append(train_loss)
            self.history_["val_loss"].append(val_loss)
            scheduler.step()

            if (epoch + 1) % 10 == 0:
                print(f"    Epoch {epoch+1:>3}/{self.epochs}  "
                      f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}")

            # Early stopping
            if val_loss < best_val_loss - 1e-5:
                best_val_loss  = val_loss
                patience_count = 0
                # Save best weights
                self._best_state = {k: v.cpu().clone()
                                    for k, v in self.model.state_dict().items()}
            else:
                patience_count += 1
                if patience_count >= self.patience:
                    print(f"    Early stopping at epoch {epoch+1}")
                    break

        # Restore best weights
        if hasattr(self, "_best_state"):
            self.model.load_state_dict(self._best_state)

        self.device_str = device_str
        self.fit_time   = time.time() - t0
        print(f"  ✓  CNN trained in {self.fit_time:.1f}s  "
              f"(best val_loss={best_val_loss:.4f})")
        return self

    def predict(self, X_patches: np.ndarray) -> np.ndarray:
        """Predict on patch tensor. Returns 1D array of predictions."""
        import torch
        from torch.utils.data import TensorDataset, DataLoader

        if self.model is None:
            raise RuntimeError("Call fit() first.")

        device = torch.device(self.device_str)
        self.model.eval()
        ds = TensorDataset(torch.tensor(X_patches, dtype=torch.float32))
        dl = DataLoader(ds, batch_size=self.batch_size, shuffle=False)

        preds = []
        with torch.no_grad():
            for (xb,) in dl:
                xb   = xb.to(device)
                pred = self.model(xb)
                preds.append(pred.cpu().numpy())
        return np.concatenate(preds)