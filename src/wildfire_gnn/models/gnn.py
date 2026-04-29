"""
GNN Architectures — Phase 5A of wildfire-uncertainty-gnn.

THREE ARCHITECTURES (for ablation study)
-----------------------------------------
1. GAT  — Graph Attention Network  (PRIMARY model)
2. GCN  — Graph Convolutional Net  (ablation: no attention)
3. GraphSAGE                       (ablation: inductive aggregation)

WHY GAT IS PRIMARY
------------------
GAT learns PER-NEIGHBOR ATTENTION WEIGHTS. For wildfire spread,
some neighbors matter more than others:
  - An upslope neighbor matters more than a downslope neighbor
  - A high-fuel neighbor matters more than a low-fuel neighbor
GAT learns these asymmetric relationships from data.
GCN uses fixed equal-weight aggregation — it cannot learn this.

WHY GAUSSIAN NLL LOSS (Gap 1)
------------------------------
FSim burn probability labels are STOCHASTIC Monte Carlo estimates.
Standard MSE treats them as noise-free ground truth — this is wrong.
Gaussian NLL loss: model outputs mean + log_variance per node.
  loss = 0.5 * (log(var) + (y - mu)^2 / var)
This forces the model to be appropriately uncertain where labels
are unreliable, directly addressing Gap 1.

WHY MC DROPOUT (Gap 2)
-----------------------
Keeping dropout ON at inference and running 20+ forward passes
gives epistemic uncertainty (model uncertainty about weights).
  predictions = [model(x) for _ in range(20)]
  mean = np.mean(predictions)  ← final prediction
  std  = np.std(predictions)   ← epistemic uncertainty
This enables calibrated uncertainty intervals — Gap 2.

ARCHITECTURE DETAILS
--------------------
  in_channels : 61  (from graph.num_node_features — Phase 3 confirmed)
  hidden      : 256
  out_channels: 2   (mean + log_variance for Gaussian NLL)
  layers      : 4
  heads       : 8   (GAT only)
  dropout     : 0.3 (also used at inference for MC Dropout)
  residual    : True
  batch_norm  : True

KNOWN FAILURE FROM PREVIOUS PROJECT
-------------------------------------
val_loss plateau ≈ 0.88 with train_loss 0.56 = generalization gap.
Solutions implemented here:
  1. Residual connections prevent vanishing gradients
  2. Batch normalization stabilizes training
  3. Early stopping on val_loss with patience=20
  4. Gradient clipping at 1.0
  5. Cosine LR schedule
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ════════════════════════════════════════════════════════════════════════════
# Shared components
# ════════════════════════════════════════════════════════════════════════════

class ResidualBlock(nn.Module):
    """
    Linear → BatchNorm → ReLU → Dropout with residual connection.
    Used in all GNN architectures between message-passing layers.
    """
    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.3):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.bn     = nn.BatchNorm1d(out_dim)
        self.drop   = nn.Dropout(dropout)
        # Projection if dimensions differ
        self.proj   = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        residual = self.proj(x)
        out      = self.drop(F.relu(self.bn(self.linear(x))))
        return out + residual


class GaussianNLLHead(nn.Module):
    """
    Dual output head: predicts mean AND log_variance per node.

    Output: (mean, log_var), both shape (N,)

    Loss (computed in GNNPipeline, not here):
      gaussian_nll = 0.5 * (log_var + (y - mean)^2 / exp(log_var))
    """
    def __init__(self, in_dim: int):
        super().__init__()
        self.mean_head    = nn.Linear(in_dim, 1)
        self.logvar_head  = nn.Linear(in_dim, 1)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        mean    = self.mean_head(x).squeeze(-1)
        log_var = self.logvar_head(x).squeeze(-1)
        # Clamp log_var for numerical stability
        log_var = torch.clamp(log_var, min=-10.0, max=10.0)
        return mean, log_var


# ════════════════════════════════════════════════════════════════════════════
# 1. GAT — Graph Attention Network (PRIMARY)
# ════════════════════════════════════════════════════════════════════════════

class GATWildfire(nn.Module):
    """
    Graph Attention Network for wildfire burn probability prediction.

    Architecture:
      Input projection → 4 GAT layers → output head
      Each GAT layer: multi-head attention + residual + BN + dropout

    Parameters
    ----------
    in_channels  : 61 (graph.num_node_features)
    hidden       : 256
    out_dim_head : 128 (pre-head dimension)
    num_layers   : 4
    heads        : 8 (attention heads)
    dropout      : 0.3 (also used for MC Dropout at inference)
    """
    def __init__(
        self,
        in_channels: int  = 61,
        hidden:      int  = 256,
        num_layers:  int  = 4,
        heads:       int  = 8,
        dropout:     float = 0.3,
    ):
        super().__init__()
        try:
            from torch_geometric.nn import GATConv
        except ImportError:
            raise ImportError("pip install torch-geometric")

        self.dropout = dropout
        self.name    = "GAT"

        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(in_channels, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # GAT layers
        self.convs = nn.ModuleList()
        self.bns   = nn.ModuleList()
        self.projs = nn.ModuleList()

        for i in range(num_layers):
            # Each GAT layer: hidden → hidden (concat heads then project)
            conv = GATConv(
                in_channels  = hidden,
                out_channels = hidden // heads,
                heads        = heads,
                dropout      = dropout,
                concat       = True,   # output = heads * (hidden//heads) = hidden
                add_self_loops = True,
            )
            self.convs.append(conv)
            self.bns.append(nn.BatchNorm1d(hidden))
            # Projection for residual (same dim here, but explicit)
            self.projs.append(nn.Identity())

        # Output head
        self.head = GaussianNLLHead(hidden)

    def forward(
        self,
        x:          Tensor,
        edge_index: Tensor,
        training:   bool = True,
    ) -> tuple[Tensor, Tensor]:
        """
        Parameters
        ----------
        x          : (N, 61) node feature matrix
        edge_index : (2, E) edge index
        training   : if False AND dropout > 0, dropout is still ON
                     for MC Dropout inference. Use model.train() to control.

        Returns
        -------
        mean    : (N,) predicted burn probability (transformed scale)
        log_var : (N,) predicted log-variance (aleatoric uncertainty)
        """
        h = self.input_proj(x)

        for conv, bn, proj in zip(self.convs, self.bns, self.projs):
            residual = proj(h)
            h        = conv(h, edge_index)
            h        = bn(h)
            h        = F.relu(h)
            h        = F.dropout(h, p=self.dropout, training=self.training)
            h        = h + residual

        mean, log_var = self.head(h)
        return mean, log_var


# ════════════════════════════════════════════════════════════════════════════
# 2. GCN — Graph Convolutional Network (ablation: no attention)
# ════════════════════════════════════════════════════════════════════════════

class GCNWildfire(nn.Module):
    """
    GCN ablation baseline. Equal-weight neighbor aggregation.
    Compare vs GAT to measure: does attention add value?
    """
    def __init__(
        self,
        in_channels: int   = 61,
        hidden:      int   = 256,
        num_layers:  int   = 4,
        dropout:     float = 0.3,
    ):
        super().__init__()
        from torch_geometric.nn import GCNConv

        self.dropout = dropout
        self.name    = "GCN"

        self.input_proj = nn.Sequential(
            nn.Linear(in_channels, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.convs = nn.ModuleList([
            GCNConv(hidden, hidden, add_self_loops=True, normalize=True)
            for _ in range(num_layers)
        ])
        self.bns = nn.ModuleList([
            nn.BatchNorm1d(hidden) for _ in range(num_layers)
        ])
        self.head = GaussianNLLHead(hidden)

    def forward(self, x: Tensor, edge_index: Tensor,
                training: bool = True) -> tuple[Tensor, Tensor]:
        h = self.input_proj(x)
        for conv, bn in zip(self.convs, self.bns):
            residual = h
            h        = conv(h, edge_index)
            h        = bn(h)
            h        = F.relu(h)
            h        = F.dropout(h, p=self.dropout, training=self.training)
            h        = h + residual
        return self.head(h)


# ════════════════════════════════════════════════════════════════════════════
# 3. GraphSAGE — Inductive aggregation (ablation)
# ════════════════════════════════════════════════════════════════════════════

class GraphSAGEWildfire(nn.Module):
    """
    GraphSAGE ablation. Mean aggregation of neighbor features.
    Inductive: can generalize to unseen nodes — key for geographic split.
    Compare vs GAT to measure: does attention over inductive baseline add value?
    """
    def __init__(
        self,
        in_channels: int   = 61,
        hidden:      int   = 256,
        num_layers:  int   = 4,
        dropout:     float = 0.3,
    ):
        super().__init__()
        from torch_geometric.nn import SAGEConv

        self.dropout = dropout
        self.name    = "GraphSAGE"

        self.input_proj = nn.Sequential(
            nn.Linear(in_channels, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.convs = nn.ModuleList([
            SAGEConv(hidden, hidden, aggr="mean", normalize=True)
            for _ in range(num_layers)
        ])
        self.bns = nn.ModuleList([
            nn.BatchNorm1d(hidden) for _ in range(num_layers)
        ])
        self.head = GaussianNLLHead(hidden)

    def forward(self, x: Tensor, edge_index: Tensor,
                training: bool = True) -> tuple[Tensor, Tensor]:
        h = self.input_proj(x)
        for conv, bn in zip(self.convs, self.bns):
            residual = h
            h        = conv(h, edge_index)
            h        = bn(h)
            h        = F.relu(h)
            h        = F.dropout(h, p=self.dropout, training=self.training)
            h        = h + residual
        return self.head(h)


# ════════════════════════════════════════════════════════════════════════════
# Factory function
# ════════════════════════════════════════════════════════════════════════════

def build_model(
    architecture: str  = "GAT",
    in_channels:  int  = 61,
    hidden:       int  = 256,
    num_layers:   int  = 4,
    heads:        int  = 8,
    dropout:      float = 0.3,
) -> nn.Module:
    """
    Build a GNN model by name.

    Parameters
    ----------
    architecture : "GAT" | "GCN" | "GraphSAGE"
    in_channels  : must match graph.num_node_features (61 after Phase 3)
    hidden       : hidden dimension
    num_layers   : number of message-passing layers
    heads        : attention heads (GAT only)
    dropout      : dropout rate (same rate used for MC Dropout at inference)

    Returns
    -------
    nn.Module with forward(x, edge_index) → (mean, log_var)
    """
    arch = architecture.upper().replace("-", "").replace("_", "")

    if arch == "GAT":
        return GATWildfire(
            in_channels=in_channels, hidden=hidden,
            num_layers=num_layers, heads=heads, dropout=dropout,
        )
    elif arch == "GCN":
        return GCNWildfire(
            in_channels=in_channels, hidden=hidden,
            num_layers=num_layers, dropout=dropout,
        )
    elif arch in ("GRAPHSAGE", "SAGE"):
        return GraphSAGEWildfire(
            in_channels=in_channels, hidden=hidden,
            num_layers=num_layers, dropout=dropout,
        )
    else:
        raise ValueError(f"Unknown architecture: {architecture}. "
                         f"Choose from: GAT, GCN, GraphSAGE")


def gaussian_nll_loss(
    mean:    Tensor,
    log_var: Tensor,
    target:  Tensor,
) -> Tensor:
    """
    Gaussian Negative Log-Likelihood loss.

    loss = 0.5 * mean(log_var + (target - mean)^2 / exp(log_var))

    This is the correct loss for Gap 1: it forces the model to be
    appropriately uncertain (high log_var) where prediction is hard,
    rather than minimising raw squared error on noisy simulation labels.

    Parameters
    ----------
    mean    : (N,) predicted mean
    log_var : (N,) predicted log-variance (clamped in model)
    target  : (N,) quantile-transformed burn probability

    Returns
    -------
    Scalar loss tensor
    """
    var  = torch.exp(log_var)
    loss = 0.5 * (log_var + (target - mean) ** 2 / (var + 1e-8))
    return loss.mean()


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)