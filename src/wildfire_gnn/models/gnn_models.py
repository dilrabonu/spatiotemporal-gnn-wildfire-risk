"""
Phase 5A — Deterministic GNN Models.

WHY THIS FILE EXISTS
--------------------
Phase 5A tests whether graph topology adds predictive value beyond:

1. Tabular ML baselines:
   - Random Forest
   - XGBoost

2. Spatial CNN baseline:
   - CNN sees local raster patches
   - CNN does NOT use graph message passing

The scientific comparison is:

    RF/XGBoost  ->  CNN  ->  GNN

RF/XGBoost:
    No spatial message passing.

CNN:
    Local convolutional spatial context, no graph topology.

GNN:
    Explicit graph topology and neighbor message passing.

MODELS IMPLEMENTED
------------------
1. GCN
   Basic spectral-style message passing.

2. GraphSAGE
   Neighborhood aggregation baseline.

3. GAT
   Attention-based neighbor weighting.
   This is the main Phase 5A model because wildfire spread is spatially uneven:
   not all neighbors should contribute equally.

OUTPUT
------
All models output one scalar per node:

    y_pred_transformed

This prediction is in transformed target space.
Metrics must be computed only after inverse-transforming predictions
back to original burn probability scale.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GNNConfig:
    """Configuration container for deterministic GNN models."""
    model_name: str = "gat"
    in_channels: int = 61
    hidden_channels: int = 64
    num_layers: int = 3
    dropout: float = 0.3
    heads: int = 4


class MLPHead(nn.Module):
    """
    Small prediction head used after graph message passing.

    Why not output directly from the final convolution?
    A small MLP head gives the model a little more non-linear capacity
    after neighbor aggregation.
    """

    def __init__(self, in_channels: int, hidden_channels: int, dropout: float):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class GCNModel(nn.Module):
    """
    Deterministic GCN baseline.

    Purpose:
    Establish whether simple graph convolution improves over non-graph baselines.

    Input:
    - x: node feature matrix, shape (N, 61)
    - edge_index: graph edges, shape (2, E)

    Output:
    - prediction for every node, shape (N,)
    """

    def __init__(
        self,
        in_channels: int = 61,
        hidden_channels: int = 64,
        num_layers: int = 3,
        dropout: float = 0.3,
    ):
        super().__init__()

        from torch_geometric.nn import GCNConv

        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")

        self.model_name = "GCN"
        self.dropout = dropout

        self.convs = nn.ModuleList()

        self.convs.append(GCNConv(in_channels, hidden_channels))

        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))

        self.head = MLPHead(
            in_channels=hidden_channels,
            hidden_channels=hidden_channels,
            dropout=dropout,
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        return self.head(x)


class GraphSAGEModel(nn.Module):
    """
    Deterministic GraphSAGE baseline.

    Purpose:
    Test neighborhood aggregation with learnable sampled-style aggregation.

    GraphSAGE is useful because it is often more stable than GAT on large graphs.
    """

    def __init__(
        self,
        in_channels: int = 61,
        hidden_channels: int = 64,
        num_layers: int = 3,
        dropout: float = 0.3,
    ):
        super().__init__()

        from torch_geometric.nn import SAGEConv

        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")

        self.model_name = "GraphSAGE"
        self.dropout = dropout

        self.convs = nn.ModuleList()

        self.convs.append(SAGEConv(in_channels, hidden_channels))

        for _ in range(num_layers - 1):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))

        self.head = MLPHead(
            in_channels=hidden_channels,
            hidden_channels=hidden_channels,
            dropout=dropout,
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        return self.head(x)


class GATModel(nn.Module):
    """
    Deterministic Graph Attention Network.

    Purpose:
    Main Phase 5A architecture.

    Why GAT?
    Wildfire spread is not isotropic. Some neighboring cells matter more than
    others depending on fuel, ignition probability, exposure, terrain, and
    local spatial structure. Attention allows the model to learn unequal
    neighbor importance.
    """

    def __init__(
        self,
        in_channels: int = 61,
        hidden_channels: int = 64,
        num_layers: int = 3,
        heads: int = 4,
        dropout: float = 0.3,
    ):
        super().__init__()

        from torch_geometric.nn import GATConv

        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")

        self.model_name = "GAT"
        self.dropout = dropout
        self.heads = heads

        self.convs = nn.ModuleList()

        self.convs.append(
            GATConv(
                in_channels=in_channels,
                out_channels=hidden_channels,
                heads=heads,
                concat=True,
                dropout=dropout,
            )
        )

        current_channels = hidden_channels * heads

        for _ in range(num_layers - 1):
            self.convs.append(
                GATConv(
                    in_channels=current_channels,
                    out_channels=hidden_channels,
                    heads=heads,
                    concat=True,
                    dropout=dropout,
                )
            )
            current_channels = hidden_channels * heads

        self.head = MLPHead(
            in_channels=current_channels,
            hidden_channels=hidden_channels,
            dropout=dropout,
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        return self.head(x)


def build_gnn_model(
    model_name: str,
    in_channels: int = 61,
    hidden_channels: int = 64,
    num_layers: int = 3,
    dropout: float = 0.3,
    heads: int = 4,
) -> nn.Module:
    """
    Build deterministic Phase 5A GNN model.

    Parameters
    ----------
    model_name:
        One of: gcn, graphsage, gat.
    in_channels:
        Number of input node features. For this project, expected = 61.
    hidden_channels:
        Hidden embedding size.
    num_layers:
        Number of graph convolution layers.
    dropout:
        Dropout probability.
    heads:
        Number of GAT attention heads.

    Returns
    -------
    torch.nn.Module
    """
    model_name = model_name.lower()

    if model_name == "gcn":
        return GCNModel(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            num_layers=num_layers,
            dropout=dropout,
        )

    if model_name == "graphsage":
        return GraphSAGEModel(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            num_layers=num_layers,
            dropout=dropout,
        )

    if model_name == "gat":
        return GATModel(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            num_layers=num_layers,
            heads=heads,
            dropout=dropout,
        )

    raise ValueError(
        f"Unknown model_name='{model_name}'. Expected one of: gcn, graphsage, gat."
    )