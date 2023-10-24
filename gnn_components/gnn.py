import torch
from torch.nn import Sequential as Seq, Linear as Lin
from torch_geometric.nn import MetaLayer
from edge_model import EdgeModel
from mlp import MLP


class GNN(torch.nn.Module):
    def __init__(self, num_edge_feats, num_node_feats, num_global_feats, num_hid_layers, size_hid_layers,
                 num_mlp_layers, size_mlp_layers, num_outputs, num_mp, device, activation=None, norm=None):
        super(GNN, self).__init__()

        self.num_edge_feats = num_edge_feats
        self.num_node_feats = num_node_feats
        self.num_global_feats = num_global_feats

        self.num_hid_layers = num_hid_layers
        self.size_hid_layers = size_hid_layers
        self.activation = activation
        self.norm = norm

        # Load MLP

        # Load Edge_model
        self.edge_model = EdgeModel(num_edge_feats, num_node_feats, num_global_feats, edge_mlp_node, edge_mlp_all,
                                    activation=None, norm=None)

        self.num_mlp_layers = num_mlp_layers
        self.size_mlp_layers = size_mlp_layers
        self.num_outputs = num_outputs

        self.num_mp = num_mp

        self.device = device



        self.meta = MetaLayer(
            EdgeModel(self.num_edge_feats, self.num_node_feats, self.num_hid_layers, self.size_hid_layers,
                      self.activation, self.norm),
            None,
            None)

        # Initialize MLP
        self.last_mlp.apply(init_weights)

    def forward(self, grph):
        # Extract all from MiniBatch graph
        x, ei, ea, btc = grph.x, grph.edge_index, grph.edge_attr, grph.batch

        # Get batch size
        batch_size = grph.y.size()[0]

        # Create empty global feature
        u = torch.full(size=(batch_size, self.num_global_feats), fill_value=0.1, dtype=torch.float).to(self.device)

        # Do message passing
        for _ in range(self.num_mp):
            x, ea, u = self.meta(x=x, edge_index=ei, edge_attr=ea, u=u, batch=btc)

        # Run MLP on output
        return self.last_mlp(u)