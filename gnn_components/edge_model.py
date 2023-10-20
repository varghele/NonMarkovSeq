import torch
from torch.nn import Sequential as Seq, Linear as Lin


class EdgeModel(torch.nn.Module):
    def __init__(self, num_edge_feats, num_node_feats, num_global_feats, num_hid_layers, size_hid_layers,
                 activation=None, norm=None):
        super(EdgeModel, self).__init__()

        self.num_edge_feats = num_edge_feats
        self.num_node_feats = num_node_feats

        self.num_hid_layers = num_hid_layers
        self.size_hid_layers = size_hid_layers
        self.activation = activation
        self.norm = norm

        self.num_inputs = self.num_edge_feats + 2 * self.num_node_feats

        # hidden = HIDDEN_EDGE
        # in_channels = HID_EDGE_ENC+2*HID_NODE_ENC

        # Set up general adjustable MLP
        # Could also do Seq([*module_list])
        if self.size_hid_layers > 0:
            self.edge_mlp = Seq()
            # Add first input layer
            self.edge_mlp.add_module(f"Lin{0}", Lin(self.num_inputs, self.size_hid_layers))
            if self.activation is not None:
                self.edge_mlp.add_module(f"Act{0}", self.activation)
            if self.norm is not None:
                self.edge_mlp.add_module(f"Norm{0}", self.norm(self.size_hid_layers))

            # Hidden layers
            for l in range(1, self.num_hid_layers):
                self.edge_mlp.add_module(f"Lin{l}", Lin(self.size_hid_layers, self.size_hid_layers))
                if self.activation is not None:
                    self.edge_mlp.add_module(f"Act{l}", self.activation)
                if self.norm is not None:
                    self.edge_mlp.add_module(f"Norm{l}", self.norm(self.size_hid_layers))
            # Add last layer
            self.edge_mlp.add_module(f"Lin{self.num_hid_layers}", Lin(self.size_hid_layers, self.num_edge_feats))
        else:
            self.edge_mlp = Seq(Lin(self.num_inputs, self.num_edge_feats))

        # Initialize MLP
        self.edge_mlp.apply(init_weights)

    def forward(self, src, dest, edge_attr, u, batch):
        # source, target: [E, F_x], where E is the number of edges.
        # edge_attr: [E, F_e]
        # u: [B, F_u], where B is the number of graphs.
        # batch: [E] with max entry B - 1.
        out = torch.cat([src, dest, edge_attr], 1)
        return self.edge_mlp(out)