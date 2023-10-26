import torch
from init import init_weights


class EdgeModel(torch.nn.Module):
    def __init__(self, num_edge_feats, num_node_feats, num_global_feats, edge_mlp_node, edge_mlp_all):
        super(EdgeModel, self).__init__()

        self.num_edge_feats = num_edge_feats
        self.num_node_feats = num_node_feats
        self.num_global_feats = num_global_feats

        self.edge_mlp_node = edge_mlp_node
        self.edge_mlp_all = edge_mlp_all

        self.num_inputs = self.num_edge_feats + 2 * self.num_node_feats

        # Initialize MLP
        self.edge_mlp_node.apply(init_weights)
        self.edge_mlp_all.apply(init_weights)

    def forward(self, src, dst, edge_attr, u, batch):
        # source, target: [E, F_x], where E is the number of edges.
        # edge_attr: [E, F_e]
        # u: [B, F_u], where B is the number of graphs.
        # batch: [E] with max entry B - 1.
        node_out = self.edge_mlp_node(src)
        out = torch.cat([node_out, edge_attr, u[batch]], 1)
        return self.edge_mlp_all(out)
