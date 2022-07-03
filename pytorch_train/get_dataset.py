import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms

trans = transforms.Compose(
                    [
                    transforms.ToTensor(),
                    transforms.Normalize(0.031,0.1115)
                    ]
)

class net(torch.nn.Module):
    def __init__(self):
        super(net,self).__init__()


