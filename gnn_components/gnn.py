import torch
from torch.nn import Sequential as Seq, Linear as Lin
from torch_geometric.nn import MetaLayer
from edge_model import EdgeModel


class GNN(torch.nn.Module):
    def __init__(self, num_edge_feats, num_node_feats, num_global_feats, num_hid_layers, size_hid_layers,
                 num_mlp_layers, size_mlp_layers, num_outputs, num_mp, device, activation=None, norm=None):
        super(GNN, self).__init__()

        self.num_edge_feats = num_edge_feats
        self.num_node_feats = num_node_feats
        self.num_global_feats = num_global_feats

        self.num_hid_layers = num_hid_layers
        self.size_hid_layers = size_hid_layers

        self.num_mlp_layers = num_mlp_layers
        self.size_mlp_layers = size_mlp_layers
        self.num_outputs = num_outputs

        self.num_mp = num_mp

        self.device = device

        self.activation = activation
        self.norm = norm

        self.meta = MetaLayer(
            EdgeModel(self.num_edge_feats, self.num_node_feats, self.num_hid_layers, self.size_hid_layers,
                      self.activation, self.norm),
            None,
            None)

        # MLP that calculates the output from the graph features
        # Could also do Seq([*module_list])
        if self.size_mlp_layers > 0:
            self.last_mlp = Seq()
            # Add first input layer
            self.last_mlp.add_module(f"Lin{0}", Lin(self.num_global_feats, self.size_mlp_layers))
            if self.activation is not None:
                self.last_mlp.add_module(f"Act{0}", self.activation)
            if self.norm is not None:
                self.last_mlp.add_module(f"Norm{0}", self.norm(self.size_mlp_layers))

            # Hidden layers
            for l in range(1, self.num_mlp_layers):
                self.last_mlp.add_module(f"Lin{l}", Lin(self.size_mlp_layers, self.size_mlp_layers))
                if self.activation is not None:
                    self.last_mlp.add_module(f"Act{l}", self.activation)
                if self.norm is not None:
                    self.last_mlp.add_module(f"Norm{l}", self.norm(self.size_mlp_layers))
            # Add last layer
            self.last_mlp.add_module(f"Lin{self.num_mlp_layers}", Lin(self.size_mlp_layers, self.num_outputs))
        else:
            self.last_mlp = Seq(Lin(self.num_global_feats, self.num_outputs))

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