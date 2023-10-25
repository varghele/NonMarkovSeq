import torch
from torch.nn import ModuleList as ModL
from gnn_components.gnn import GNN
from torch_geometric.data import Data


class RecEdgeGnn(torch.nn.Module):
    def __init__(self, num_blocks, list_of_node_feats):
        super(RecEdgeGnn, self).__init__()

        # Establish model as Sequential
        self.recurrent_net = ModL()

        # Define number of blocks that (e.g. number of machines in line) for model
        self.num_blocks = num_blocks

        # Add independent GNN blocks to model
        for k in range(self.num_blocks):
            self.recurrent_net.add_module(f"GNN{k}", GNN())

    def forward(self, grph):

        # get the batch size of the current batch (important to allocate the right nodes)
        bs = grph.batch.size()[0]

        # iterate over every machine in the line
        for k in range(1, self.num_blocks+1):
            # find multiples of k that are within batch size for allocation list
            k_alloc = [i+k for i in range(bs) if ((i%self.num_blocks)==0)]
            # create a sub-graph of the batch that only takes the k-th node
            sub_x = grph.x[k_alloc, :]
            # per node exist exactly two edges (in/out), so we simply take k_alloc*2
            k2_alloc = [i*2 for i in k_alloc]
            sub_ea = grph.ea[k2_alloc, :]
            sub_ei = grph.ei[:, k2_alloc]
            # globals are the same for each node (within batch)
            sub_y = grph.y

            sub_grph = Data()

            out = self.recurrent_net(inc)

        return None
