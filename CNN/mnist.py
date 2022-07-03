import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torchvision
from torchvision import transforms


batch_size = 64
transform = transforms.Compose([
    transforms.ToTensor(),   #调成形状为C*W*H  取值为【0，1】
    transforms.Normalize((0.1307,),(0.3081,))    #均值，标准差
])

train_set = torchvision.datasets.MNIST(root='./dataset/mnist',train=True,download=True,transform=transform)
train_loader = DataLoader(train_set,shuffle=True,batch_size=batch_size)
text_set = torchvision.datasets.MNIST(root='./dataset/mnist',train=False,download=True,transform=transform)
test_loader = DataLoader(text_set,shuffle=False,batch_size=batch_size)


#卷积神经网络
class CNN(torch.nn.Module):
    def __init__(self):
        super(CNN,self).__init__()
        self.conv1 = torch.nn.Conv2d(1,10,kernel_size=5)
        self.conv2 = torch.nn.Conv2d(10,20,kernel_size=5)
        self.pooling = torch.nn.MaxPool2d(2)
        self.fc = torch.nn.Linear(320,10)
        self.relu = torch.nn.ReLU()

    def forward(self,x):
        batch_size = x.size(0)
        x = self.relu(self.pooling(self.conv1(x)))
        x = self.relu(self.pooling(self.conv2(x)))
        x = x.view(batch_size,-1)
        return self.fc(x)




#
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.relu = torch.nn.ReLU()
        self.linear1 = torch.nn.Linear(784,512)
        self.linear2 = torch.nn.Linear(512,256)
        self.linear3 = torch.nn.Linear(256,128)
        self.linear4 = torch.nn.Linear(128,64)
        self.linear5 = torch.nn.Linear(64,10)

    def forward(self,x):
        x = x.view(-1,784)
        x = self.relu(self.linear1(x))
        x = self.relu(self.linear2(x))
        x = self.relu(self.linear3(x))
        x = self.relu(self.linear4(x))
        return self.linear5(x)

# model = Model()
model = CNN()

#使用GPU运算
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model.to(device)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(),lr=0.01,momentum=0.5)    #momentum 表示冲量

# for epoch in range(100):
#     for i,data in enumerate(train_loader,0):
#         input,label = data
#         output_linear = model(input)
#         loss = criterion(output_linear,label)
#         print(epoch,i,loss.data)
#
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()

def train(epoch):
    running_loss = 0.0
    for batch_idx,data in enumerate(train_loader,0):
        inputs,labels = data


        inputs,labels = inputs.to(device),labels.to(device)   #再GPU上运算

        outputs_linear = model(inputs)
        loss = criterion(outputs_linear,labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        if batch_idx % 300 == 299:
            print(epoch+1,batch_idx+1,'loss=',running_loss/300)
            running_loss = 0.0

def test():
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            inputs,labels = data
            print(inputs.size())
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            _,pred = torch.max(outputs.data,dim=1)  #_为outputs.data的最大值，pred为最大值的下标
            total += labels.size(0)
            correct += (pred == labels).sum().item()

        print('正确率为：',correct/total*100,'%')


if __name__=='__main__':
    for epoch in range(1):
        train(epoch)
        test()
