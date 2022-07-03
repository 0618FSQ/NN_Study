import matplotlib.pyplot as plt
import torch


x_data = torch.Tensor([[1.0],[2.0],[3.0]])
y_data = torch.Tensor([[2.0],[4.0],[6.0]])



class LinearModel(torch.nn.Module):
    def __init__(self):
        super(LinearModel,self).__init__()
        self.linear = torch.nn.Linear(1,1)   #torch.nn.Linear(in_features,out_features,bias=True)  包含了两个张量w和b

    def forward(self,x):
        y_pred = self.linear(x)
        return y_pred


model = LinearModel()

criterion = torch.nn.MSELoss(size_average=False)       
optimizer = torch.optim.SGD(model.parameters(),lr=0.01)
costs_SGD = []
for epoch in range(1000):

    y_pred = model(x_data)
    loss = criterion(y_pred,y_data)
    costs_SGD.append(loss)
    print(epoch,loss.item())

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

print('w=',model.linear.weight.item())
print('b=',model.linear.bias.item())



x_test = torch.Tensor([4.0])
y_test = model(x_test)
print('y_pred:',4,y_test.data.item())


plt.figure()
plt.plot(costs_SGD)
plt.show()



# print(model.linear.parameters())

