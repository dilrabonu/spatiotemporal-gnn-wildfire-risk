"""
Phase 5A — Deterministic GNN Models

Implements:
- GCN
- GraphSAGE
- GAT

All models output:
    y_pred (regression)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv, GATConv


class GCNModel(nn.Module):
    def __init__(self, in_channels, hidden_channels=64, num_layers=3, dropout=0.2):
        super().__init__()

        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels))

        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))

        self.convs.append(GCNConv(hidden_channels, 1))
        self.dropout = dropout

    def forward(self, x, edge_index):
        for conv in self.convs[:-1]:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.convs[-1](x, edge_index)
        return x.squeeze(-1)


class GraphSAGEModel(nn.Module):
    def __init__(self, in_channels, hidden_channels=64, num_layers=3, dropout=0.2):
        super().__init__()

        self.convs = nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels))

        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))

        self.convs.append(SAGEConv(hidden_channels, 1))
        self.dropout = dropout

    def forward(self, x, edge_index):
        for conv in self.convs[:-1]:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.convs[-1](x, edge_index)
        return x.squeeze(-1)


class GATModel(nn.Module):
    def __init__(self, in_channels, hidden_channels=64, heads=4, dropout=0.2):
        super().__init__()

        self.gat1 = GATConv(in_channels, hidden_channels, heads=heads)
        self.gat2 = GATConv(hidden_channels * heads, hidden_channels, heads=1)
        self.out = nn.Linear(hidden_channels, 1)

        self.dropout = dropout

    def forward(self, x, edge_index):
        x = self.gat1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.gat2(x, edge_index)
        x = F.elu(x)

        x = self.out(x)
        return x.squeeze(-1)