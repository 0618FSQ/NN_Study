# %%

import torch
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree


class GCNConv(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super().__init__(aggr='add')  # "Add" aggregation (Step 5).
        self.lin = torch.nn.Linear(in_channels, out_channels)

    def forward(self, x, edge_index):
        # x has shape [N, in_channels]
        # edge_index has shape [2, E]

        # Step 1: Add self-loops to the adjacency matrix.
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        print("\nAdd self-loops edge_index--------------")
        print(edge_index)

        # Step 2: Linearly transform node feature matrix.
        x = self.lin(x)
        print("\nLinearly transform--------------")
        print(x)

        # Step 3: Compute normalization.
        row, col = edge_index
        print("\nrow-------", row)
        print("\ncol-------", col)
        deg = degree(col, x.size(0), dtype=x.dtype)
        print("\ndegree------------------")
        print(deg)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
        print("\nnorm--------------")
        print(norm)

        # Step 4-5: Start propagating messages.
        return self.propagate(edge_index, x=x, norm=norm)



    def message(self, x_j, norm):
        # x_j has shape [E, out_channels]

        print("\nx_j--------------")
        print(x_j)
        print("\nnorm.view(-1, 1) * x_j--------------")
        print(norm.view(-1, 1) * x_j)

        # Step 4: Normalize node features.
        return norm.view(-1, 1) * x_j

# 定义Data实例的属性数据
edge_index = torch.tensor([[1, 2, 3], [0, 0, 0]], dtype=torch.long)
x = torch.tensor([[1], [2], [3], [4]], dtype=torch.float)

conv = GCNConv(1, 2)
ret = conv(x, edge_index)
ret
print(ret)

