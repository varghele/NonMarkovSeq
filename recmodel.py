import torch.nn

from gnn_components.gnn import GNN


class RecEdgeGnn(torch.nn.Module):
    def __init__(self):
        super(RecEdgeGnn)