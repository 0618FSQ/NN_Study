import torch
import networkx as nx
from torch_geometric.data import InMemoryDataset,download_url
from torch_geometric.datasets import KarateClub
from torch_geometric.utils import to_networkx
import matplotlib.pyplot as plt


def visualize_grah(G,color):
    plt.figure(figsize=(7,7))
    plt.xticks([])
    plt.yticks([])
    nx.draw_networkx(G,pos=nx.spring_layout(G,seed=42),with_labels=False,node_color = color,cmap = 'Set2')
    plt.show()

def visualize_embedding(h,color,epoch=None,loss=None):
    plt.figure(figsize=(7,7))
    plt.xticks([])
    plt.yticks([])
    h = h.detach().cpu().numpy()
    plt.scatter(h[:,0],h[:,1],s=140,c = color,cmap='Set2')
    if epoch is not None and loss is not None:
        plt.xlabel(f'epochL{epoch},Loss:{loss.item():.4f}',fontsize=16)
    plt.show()

# class Amazon(InMemoryDataset):
#     url = 'https://github.com/shchur/gnn-benchmark/raw/master/data/npz/'
#
#     def __init__(self,root,name,transform=None,pre_transform=None):
#         self.name = name.lower()
#         assert self.name in ['computers','photo']
#         super(Amazon,self).__init__(root,transform,pre_transform)
#         self.data,self.slices = torch.load(self.processed_path[0])

dataset = KarateClub()
print(f'Datasets;{dataset}:')
print(len(dataset))
print(dataset.num_node_features)
print(dataset.num_classes)


data = dataset[0]
print(data)
print(data.num_nodes)


# edge_index = data.edge_index
# print(edge_index.t())
#
# G = to_networkx(data,to_undirected=True)
# visualize_grah(G,color=data.y)
#
#


