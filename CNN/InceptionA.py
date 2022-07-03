import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F
import matplotlib.pyplot as plt

transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.1307,),(0.3081,))
    ]
)


train_set = torchvision.datasets.MNIST(root='./dataset/mnist',train=True,download=True,transform=transform)
test_set = torchvision.datasets.MNIST(root='./dataset/mnist',train=False,download=True,transform=transform)


train_loader = DataLoader(dataset=train_set,batch_size=64,shuffle=True)
test_loader = DataLoader(dataset=test_set,batch_size=64,shuffle=False)

class ResidualBlock(torch.nn.Module):
    def __init__(self,channel):
        super(ResidualBlock,self).__init__()
        self.conv1 = torch.nn.Conv2d(channel,channel,kernel_size=3,padding=1)
        self.conv2 = torch.nn.Conv2d(channel,channel,kernel_size=3,padding=1)

    def forward(self,x):
        y = F.relu(self.conv1(x))
        y = self.conv2(y)
        return F.relu(y+x)


class InceptionA(torch.nn.Module):
    def __init__(self,in_channels):
        super(InceptionA,self).__init__()
        self.batch_pool = torch.nn.Conv2d(in_channels,24,kernel_size=(1,1))

        self.branch1x1 = torch.nn.Conv2d(in_channels,16,kernel_size=(1,1))

        self.branch5x5_1 = torch.nn.Conv2d(in_channels,16,kernel_size=(1,1))
        self.branch5x5_2 = torch.nn.Conv2d(16,24,kernel_size=(5,5),padding=2)

        self.branch3x3_1 = torch.nn.Conv2d(in_channels,16,kernel_size=(1,1))
        self.branch3x3_2 = torch.nn.Conv2d(16,24,kernel_size=(3,3),padding=1)
        self.branch3x3_3 = torch.nn.Conv2d(24,24,kernel_size=(3,3),padding=1)

    def forward(self,x):
        batch_pool = F.avg_pool2d(x,kernel_size=3,stride=1,padding=1)
        batch_pool = self.batch_pool(batch_pool)

        branch1x1 = self.branch1x1(x)

        branch5x5 = self.branch5x5_1(x)
        branch5x5 = self.branch5x5_2(branch5x5)

        branch3x3 = self.branch3x3_1(x)
        branch3x3 = self.branch3x3_2(branch3x3)
        branch3x3 = self.branch3x3_3(branch3x3)

        outputs = [batch_pool,branch1x1,branch5x5,branch3x3]
        outputs = torch.cat(outputs,dim=1)
        return outputs


class Net_Residual(torch.nn.Module):
    def __init__(self):
        super(Net_Residual,self).__init__()
        self.mp = torch.nn.MaxPool2d(2)
        self.conv1 = torch.nn.Conv2d(1,16,kernel_size=5)
        self.conv2 = torch.nn.Conv2d(16,32,kernel_size=5)

        self.rblock1 = ResidualBlock(channel=16)
        self.rblock2 = ResidualBlock(channel=32)

        self.linear = torch.nn.Linear(512,10)

    def forward(self,x):
        batch_size = x.size(0)
        x = self.mp(F.relu(self.conv1(x)))
        x = self.rblock1(x)
        x = self.mp(F.relu(self.conv2(x)))
        x = self.rblock2(x)
        x = x.view(batch_size,-1)
        x = self.linear(x)

        return x


class Net(torch.nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.conv1 = torch.nn.Conv2d(1,10,kernel_size=(5,5))
        self.conv2 = torch.nn.Conv2d(88,20,kernel_size=(5,5))

        self.incep1 = InceptionA(in_channels=10)
        self.incep2 = InceptionA(in_channels=20)

        self.linear = torch.nn.Linear(1408,10)

        self.pooling = torch.nn.MaxPool2d(2)


    def forward(self,x):
        batch_size = x.size(0)
        x = self.pooling(F.relu(self.conv1(x)))
        x = self.incep1(x)
        x = self.pooling(F.relu(self.conv2(x)))
        x = self.incep2(x)

        x = x.view(batch_size,-1)
        x = self.linear(x)

        return x


model = Net_Residual()

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(),lr=0.01,momentum=0.5)

def train(epoch):
    running_loss = 0.0
    for batch_idx,data in enumerate(train_loader,0):
        inputs,labels = data
        outputs = model(inputs)
        loss = criterion(outputs,labels)
        running_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch_idx % 300 == 299:
            print(epoch+1,batch_idx+1,'loss=',running_loss/300)
            running_loss = 0.0

def test():
    total = 0
    correct = 0
    for data in test_loader:
        inputs,labels = data
        outputs = model(inputs)

        _,pred = torch.max(outputs,dim=1)
        total += inputs.size(0)
        correct += (pred == labels).sum().item()

        accuracy = correct/total * 100
    print('accuracy:',accuracy,'%')
    return accuracy


Rates = []
for epoch in range(10):

    train(epoch)
    accuracy = test()
    Rates.append(accuracy)



plt.figure()
plt.plot(Rates)
plt.show()


# test code
# for data in train_loader:
#     input,label = data
#     output = model(input)
#     print(output.size())
#     break;








