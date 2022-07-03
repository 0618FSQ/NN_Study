import torch
from torch.nn import Linear
from torch_geometric.nn import GCNConv
from torch_geometric.datasets import KarateClub
import networkx as nx
import matplotlib.pyplot as plt
import time


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



dataset = KarateClub()
data = dataset[0]

class GCN(torch.nn.Module):
    def __init__(self):
        super(GCN,self).__init__()
        torch.manual_seed(1234)
        self.conv1 = GCNConv(dataset.num_features, 4)
        self.conv2 = GCNConv(4, 4)
        self.conv3 = GCNConv(4, 2)
        self.classifier = Linear(2, dataset.num_classes)

    def forward(self, x, edge_index):
        h = self.conv1(x, edge_index)
        h = h.tanh()
        h = self.conv2(h, edge_index)
        h = h.tanh()
        h = self.conv3(h, edge_index)
        h = h.tanh()

        out = self.classifier(h)

        return out, h

model = GCN()
# print(model)

# _,h = model(data.x,data.edge_index)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),lr=0.01)

def train(data):
    optimizer.zero_grad()
    out,h = model(data.x,data.edge_index)
    loss = criterion(out[data.train_mask],data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss,h

for epoch in range(401):
    loss,h = train(data)
    if epoch %10 ==0:
        visualize_embedding(h,color=data.y,epoch=epoch,loss=loss)
        time.sleep(0.3)

