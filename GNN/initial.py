import torch
from torch_geometric.data import Data

x = torch.tensor([[2,1],[5,6],[12,0],[3,7]],dtype=torch.float32)

y = torch.tensor([0,1,1,0],dtype=torch.float32)

edge_index = torch.tensor([
    [0,1,0,2,3],
    [1,0,3,1,2]
],dtype=torch.long)

data = Data(x=x,y=y,edge_index=edge_index)

print(data)
print(data.num_nodes)
print(data.num_edges)
print(data.num_node_features)

