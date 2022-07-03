import matplotlib.pyplot as plt
import torch
import torchvision
import torch.nn.functional as F



# train_set = torchvision.datasets.MNIST(root = r'./dataset/mnist',train=True,download=True)
# test_set = torchvision.datasets.MNIST(root = r'./dataset/mnist',train=False,download=True)

x_data = torch.Tensor([[1.0],[2.0],[3.0]])
y_data = torch.Tensor([[0],[0],[1]])

#####################################################
class LogisticRegressionModel(torch.nn.Module):
    def __init__(self):
        super(LogisticRegressionModel,self).__init__()
        self.linear = torch.nn.Linear(1,1)

    def forward(self,x):
        y_pred = F.sigmoid(self.linear(x))
        return y_pred

model = LogisticRegressionModel()

##########################################
criterion = torch.nn.BCELoss(size_average=False)
optimizer = torch.optim.SGD(model.parameters(),lr=0.01)
############################################
costs = []
iterateNum = 1000
for epoch in range(iterateNum):
    y_pred = model(x_data)
    loss = criterion(y_pred,y_data)
    costs.append(loss)
    print('epoch:',epoch,'loss=',loss.item())

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

plt.figure()
plt.plot(costs)
plt.show()

