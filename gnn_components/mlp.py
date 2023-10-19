import torch
from torch.nn import Sequential as Seq, Linear as Lin
from torch_geometric.nn import MetaLayer
from torch_geometric.utils import scatter

class MLP():