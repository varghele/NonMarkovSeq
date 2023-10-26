import torch
from torch.nn import ModuleList as ModL
from components.mlp import MLP


class NonMarkov(torch.nn.Module):
    def __init__(self, num_blocks, num_inputs, num_outputs, num_hid_layers, size_hid_layers, activation=None,
                 normalization=None):
        super(NonMarkov, self).__init__()

        # Establish model as Sequential
        self.NonMarkov = ModL()

        # Define number of blocks that (e.g. number of machines in line) for model
        self.num_blocks = num_blocks

        # Add independent GNN blocks to model
        for k in range(1,self.num_blocks+1):
            self.NonMarkov.add_module(f"MLP{k}", MLP(num_inputs, num_outputs, num_hid_layers, size_hid_layers,
                                                     activation, normalization))

    def forward(self, feats, splits):

        # establish list of outputs
        outs = []

        # get global features
        glob = feats[splits[-1]:]

        # iterate over every machine in the line
        for k in range(self.num_blocks):
            # get split features
            x = feats[splits[k:k+1]]
            # Concatenate outputs with global features
            x = torch.cat([x, glob])
            # Call the k-th block
            x = self.NonMarkov[k](x)
            # Append the output to outs
            outs.append(x.copy())

        return outs

