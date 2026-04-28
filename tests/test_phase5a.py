import torch
from wildfire_gnn.models.gnn_models import GCNModel


def test_gcn_forward():
    model = GCNModel(in_channels=10)

    x = torch.randn(100, 10)
    edge_index = torch.randint(0, 100, (2, 500))

    out = model(x, edge_index)

    assert out.shape[0] == 100