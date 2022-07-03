import pandas as pd
import torch
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

dataset_test = pd.read_csv('../dataset/otto-group-product-classification-challenge/test.csv')
x_data_test = dataset_test.values[:,1:]

y_data_test = pd.read_csv('../dataset/otto-group-product-classification-challenge/sampleSubmission.csv').values[:, 1:]
y_data_test = np.array([np.argmax(i) for i in y_data_test])+1

x_data_test = torch.from_numpy(x_data_test.astype(np.float32))
y_data_test = torch.LongTensor(y_data_test)

# print(y_data_test)
# print(y_data_test.shape)
# print(x_data_test.shape)

class DiabetesDataset(Dataset):
    def __init__(self,filepath):
        data = pd.read_csv(filepath)
        self.x_data = torch.from_numpy(data.values[:, 1:-1].astype(np.float32))
        train_labels = data.values[:, -1]
        self.y_data = torch.LongTensor(LabelEncoder().fit_transform(train_labels).astype(np.float32))      #能记一下 LabelEncoder().fit_transform(inputs).astype(int) 提取numpy数组中的数字
        self.len = self.y_data.shape[0]
    def __getitem__(self, item):
        return self.x_data[item],self.y_data[item]
    def __len__(self):
        return self.len

dataset = DiabetesDataset('../dataset/otto-group-product-classification-challenge/train.csv')
train_loader = DataLoader(dataset=dataset,batch_size=32,shuffle=True)

class Model(torch.nn.Module):
    def __init__(self):
        super(Model,self).__init__()
        self.Relu = torch.nn.ReLU()
        self.linear1 = torch.nn.Linear(93,64)
        self.linear2 = torch.nn.Linear(64,32)
        self.linear3 = torch.nn.Linear(32,16)
        self.linear4 = torch.nn.Linear(16,9)

    def forward(self,x):
        x = self.Relu(self.linear1(x))
        x = self.Relu(self.linear2(x))
        x = self.Relu(self.linear3(x))
        return self.linear4(x)

model = Model()


criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(),lr=0.01,momentum=0.5)


def train(epoch):
    loss_total = 0.0
    running_loss = 0.0
    for batch_idx,data in enumerate(train_loader,0):
        inputs,labels = data
        outputs_linear = model(inputs)
        loss = criterion(outputs_linear,labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        loss_total = loss.item()

        if batch_idx % 200 == 199:
            print(epoch+1,batch_idx+1,'loss=',running_loss/200)
            running_loss = 0.0
    return loss_total



def test():
    # CorrectRates = []
    with torch.no_grad():
        y_pred = model(x_data_test)
        _,pred = torch.max(y_pred,dim=1)
        total = x_data_test.shape[0]
        correct = (pred == y_data_test).sum().item()
        # CorrectRate = correct/total*100
        print('Correct Rate:',correct/total*100,'%')

costs = []
for epoch in range(100):
    loss = train(epoch)
    costs.append(loss)
    test()



plt.figure()
plt.plot(costs)
plt.show()



