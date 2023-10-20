import torch
from torch.nn import Sequential as Seq, Linear as Lin


class MLP(torch.nn.module):
    def __init__(self, num_inputs, num_outputs, num_hid_layers, size_hid_layers, activation=None,
                 normalization=None):
        super(MLP, self).__init__()

        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.num_hid_layers = num_hid_layers
        self.size_hid_layers = size_hid_layers
        self.activation = activation
        self.normalization = normalization

        # Set up general adjustable MLP
        # Could also do Seq([*module_list])
        if self.size_hid_layers > 0:
            self.mlp = Seq()
            # Add first input layer
            self.mlp.add_module(f"Lin{0}", Lin(self.num_inputs, self.size_hid_layers))
            if self.activation is not None:
                self.mlp.add_module(f"Act{0}", self.activation)
            if self.norm is not None:
                self.mlp.add_module(f"Norm{0}", self.normalization(self.size_hid_layers))

            # Hidden layers
            for l in range(1, self.num_hid_layers):
                self.mlp.add_module(f"Lin{l}", Lin(self.size_hid_layers, self.size_hid_layers))
                if self.activation is not None:
                    self.mlp.add_module(f"Act{l}", self.activation)
                if self.norm is not None:
                    self.mlp.add_module(f"Norm{l}", self.normalization(self.size_hid_layers))
            # Add last layer
            self.mlp.add_module(f"Lin{self.num_hid_layers}", Lin(self.size_hid_layers, self.num_outputs))
        else:
            self.mlp = Seq()
            if self.activation is not None:
                self.mlp = Seq(Lin(self.num_inputs, self.num_outputs))
            if self.norm is not None:
                self.mlp.add_module(f"Norm{0}", self.normalization(self.num_outputs))

    def forward(self, x):
        return self.mlp(x)
