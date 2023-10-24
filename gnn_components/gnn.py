import torch
from torch.nn import LeakyReLU, BatchNorm1d
from torch_geometric.nn import MetaLayer
from edge_model import EdgeModel
from mlp import MLP


class GNN(torch.nn.Module):
    def __init__(self, num_edge_feats, num_node_feats, num_global_feats, num_mlp_layers, size_mlp_layers, num_mp,
                 device, activation=LeakyReLU, normalization=BatchNorm1d):
        super(GNN, self).__init__()

        # Assigning graph features
        self.num_edge_feats = num_edge_feats
        self.num_node_feats = num_node_feats
        self.num_global_feats = num_global_feats

        # Assigning MLP definers
        self.num_hid_layers = num_mlp_layers
        self.size_hid_layers = size_mlp_layers

        # Create functions
        self.activation = activation
        self.normalization = normalization

        # Create MLPs
        self.edge_ml_node = MLP(self.num_node_feats, self.num_node_feats, self.num_hid_layers, self.size_hid_layers,
                                activation=self.activation, normalization=self.normalization)
        self.edge_mlp_all = MLP(self.num_edge_feats+self.num_edge_feats+self.num_global_feats, self.num_edge_feats,
                                self.num_hid_layers, self.size_hid_layers, activation=self.activation,
                                normalization=self.normalization)

        # Load Edge_model
        self.edge_model = EdgeModel(num_edge_feats, num_node_feats, num_global_feats, self.edge_mlp_node,
                                    self.edge_mlp_all)

        # Message-passing steps
        self.num_mp = num_mp

        # Device (GPU/CPU)
        self.device = device

        # Define Metalayer (is the GNN block)
        self.meta = MetaLayer(self.edge_model, None, None)

    def forward(self, grph):
        # Extract all from MiniBatch graph
        x, ei, ea, u, btc = grph.x, grph.edge_index, grph.edge_attr, grph.y, grph.batch

        # Do message passing
        for _ in range(self.num_mp):
            x, ea, u = self.meta(x=x, edge_index=ei, edge_attr=ea, u=u, batch=btc)

        # re-write info on graph
        grph.edge_attr = ea

        # Run MLP on output
        return grph