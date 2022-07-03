import torch

a = torch.tensor([[1,2,3],
                  [4,5,6]])
_,pred = torch.max(a,dim=1)
print(pred)