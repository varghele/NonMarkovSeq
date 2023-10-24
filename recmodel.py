import torch
from torch.nn import Sequential as Seq
from gnn_components.gnn import GNN


class RecEdgeGnn(torch.nn.Module):
    def __init__(self, num_blocks, list_of_node_feats):
        super(RecEdgeGnn, self).__init__()

        self.recurrent_net = Seq()

        self.num_blocks = num_blocks

        for b in range(self.num_blocks):
            self.recurrent_net.add_module(f"GNN{b}", GNN())

    def forward(self, grph):

        out = self.recurrent_net(inc)

        return None
