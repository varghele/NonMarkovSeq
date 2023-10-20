import torch
from torch.nn import Sequential as Seq, Linear as Lin


class MLP(torch.nn.module):
    def __init__(self, num_inputs, num_outputs, num_hidden_layers, num_hidden_units, activation=None,
                 normalization=None):
        super(MLP, self).__init__()

        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.num_hidden_layers = num_hidden_layers
        self.num_hidden_units = num_hidden_units
        self.activation = activation
        self.normalization = normalization

        # Set up general adjustable MLP
        # Could also do Seq([*module_list])
        if self.num_hidden_units > 0:
            self.mlp = Seq()
            # Add first input layer
            self.mlp.add_module(f"Lin{0}", Lin(self.num_inputs, self.num_hidden_units))
            if self.activation is not None:
                self.mlp.add_module(f"Act{0}", self.activation)
            if self.norm is not None:
                self.mlp.add_module(f"Norm{0}", self.normalization(self.num_hidden_units))

            # Hidden layers
            for l in range(1, self.num_hidden_layers):
                self.mlp.add_module(f"Lin{l}", Lin(self.num_hidden_units, self.num_hidden_units))
                if self.activation is not None:
                    self.mlp.add_module(f"Act{l}", self.activation)
                if self.norm is not None:
                    self.mlp.add_module(f"Norm{l}", self.normalization(self.num_hidden_units))
            # Add last layer
            self.mlp.add_module(f"Lin{self.num_hid_layers}", Lin(self.num_hidden_units, self.num_outputs))
        else:
            self.mlp = Seq()
            if self.activation is not None:
                self.mlp = Seq(Lin(self.num_inputs, self.num_outputs))
            if self.norm is not None:
                self.mlp.add_module(f"Norm{0}", self.normalization(self.num_outputs))

    def forward(self, x):
        return self.mlp(x)
