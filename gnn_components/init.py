import torch
from torch.nn import Sequential as Seq, Linear as Lin
from torch_geometric.nn import MetaLayer
from torch_geometric.utils import scatter


def init_weights(m):
    if type(m) == torch.nn.Linear:
        torch.nn.init.xavier_normal_(m.weight)
        m.bias.data.fill_(0.01)