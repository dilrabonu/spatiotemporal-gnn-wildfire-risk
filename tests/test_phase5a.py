"""
Tests for Phase 5A deterministic GNN models.

Run:
    pytest tests/test_phase5a.py -q
"""

from pathlib import Path
import sys

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from wildfire_gnn.models.gnn_models import (
    GCNModel,
    GraphSAGEModel,
    GATModel,
    build_gnn_model,
)


def make_tiny_graph(num_nodes: int = 20, num_features: int = 61):
    x = torch.randn(num_nodes, num_features)

    edge_index = torch.tensor(
        [
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        ],
        dtype=torch.long,
    )

    return x, edge_index


def test_gcn_forward_pass():
    x, edge_index = make_tiny_graph()

    model = GCNModel(
        in_channels=61,
        hidden_channels=16,
        num_layers=2,
        dropout=0.1,
    )

    y = model(x, edge_index)

    assert y.shape == (20,)
    assert torch.isfinite(y).all()


def test_graphsage_forward_pass():
    x, edge_index = make_tiny_graph()

    model = GraphSAGEModel(
        in_channels=61,
        hidden_channels=16,
        num_layers=2,
        dropout=0.1,
    )

    y = model(x, edge_index)

    assert y.shape == (20,)
    assert torch.isfinite(y).all()


def test_gat_forward_pass():
    x, edge_index = make_tiny_graph()

    model = GATModel(
        in_channels=61,
        hidden_channels=8,
        num_layers=2,
        heads=2,
        dropout=0.1,
    )

    y = model(x, edge_index)

    assert y.shape == (20,)
    assert torch.isfinite(y).all()


def test_build_gnn_model_factory():
    for name in ["gcn", "graphsage", "gat"]:
        model = build_gnn_model(
            model_name=name,
            in_channels=61,
            hidden_channels=16,
            num_layers=2,
            dropout=0.1,
            heads=2,
        )

        assert model is not None


def test_phase5a_graph_exists():
    graph_path = PROJECT_ROOT / "data" / "processed" / "graph_data_enriched.pt"
    assert graph_path.exists(), "Missing graph_data_enriched.pt"


def test_phase5a_output_dirs_creatable():
    for rel in [
        "reports/tables",
        "reports/figures",
        "reports/predictions",
        "reports/checkpoints",
    ]:
        path = PROJECT_ROOT / rel
        path.mkdir(parents=True, exist_ok=True)
        assert path.exists()