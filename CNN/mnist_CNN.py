import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms



transform = transforms.Compose([
    transforms.ToTensor(),   #调成形状为C*W*H  取值为【0，1】
    transforms.Normalize((0.1307,),(0.3081,))    #均值，标准差
])

train_data = torchvision.datasets.MNIST(root='./dataset/mnist',train=True,download=True,transform=transform)
test_data = torchvision.datasets.MNIST(root='./dataset/mnist',train=False,download=True,transform=transform)

train_loader = DataLoader(dataset=train_data,batch_size=32,shuffle=True)
test_loader = DataLoader(dataset=test_data,batch_size=32,shuffle=False)

class CNN(torch.nn.Module):
    def __init__(self):
        super(CNN,self).__init__()
        self.relu = torch.nn.ReLU()
        self.conv1 = torch.nn.Conv2d(1,10,kernel_size=5,bias=False)
        self.conv2 = torch.nn.Conv2d(10,20,kernel_size=5,padding=2,bias=False)
        self.conv3 = torch.nn.Conv2d(20,30,kernel_size=5,padding=2,bias=False)
        self.pooling = torch.nn.MaxPool2d(2)
        self.linear1 = torch.nn.Linear(270,128)
        self.linear2 = torch.nn.Linear(128,64)
        self.linear3 = torch.nn.Linear(64,10)


    def forward(self,x):
        batch_size = x.size(0)
        x = self.relu(self.pooling(self.conv1(x)))
        x = self.relu(self.pooling(self.conv2(x)))
        x = self.relu(self.pooling(self.conv3(x)))
        x = x.view(batch_size,-1)
        x = self.linear1(x)
        x = self.linear2(x)
        x = self.linear3(x)
        return x


model = CNN()

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model.to(device)


cirterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(),lr=0.01,momentum=0.5)


def train(epoch):
    running_loss = 0.0
    for batch_idx,data in enumerate(train_loader,0):
        inputs,labels = data

        inputs,labels = inputs.to(device),labels.to(device)

        outputs = model(inputs)
        loss = cirterion(outputs,labels)
        running_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch_idx % 300 == 299:
            print(epoch+1, batch_idx+1, 'loss=', running_loss/300)
            running_loss = 0

def test():
    total = 0
    correct = 0
    with torch.no_grad():
        for i,data in enumerate(test_loader,0):
            inputs,labels = data
            inputs,labels = inputs.to(device),labels.to(device)
            outputs = model(inputs)

            _,pred = torch.max(outputs.data,dim=1)

            total += inputs.data.size(0)
            correct += (pred == labels).sum().item()

        print('Correct Rates:',correct/total*100,'%')

if __name__ == '__main__':
    for epoch in range(4):
        train(epoch)
        test()





