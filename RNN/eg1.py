import torch

input_size = 4
hidden_size = 4
batch_size = 1
embedding_size = 5
num_class = 4

idx2char = ['e','h','l','o']
# x_data = [1,0,2,2,3]
x_data = [[1,0,2,2,3]]
y_data = [3,1,2,3,2]

one_hot_lookup = [[1,0,0,0],
                  [0,1,0,0],
                  [0,0,1,0],
                  [0,0,0,1]]
# x_one_hot = [one_hot_lookup[x] for x in x_data]

# inputs = torch.Tensor(x_one_hot).view(-1,batch_size,input_size)
inputs = torch.LongTensor(x_data)
labels = torch.LongTensor(y_data)


class model(torch.nn.Module):
    def __init__(self,input_size,hidden_size,batch_size,embedding_size,num_layers=1):
        super(model,self).__init__()

        self.embedding_size = embedding_size
        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.batch_size = batch_size

        self.embedding = torch.nn.Embedding(self.input_size,self.embedding_size)

        self.rnn = torch.nn.RNN(num_layers=self.num_layers,
                                input_size=self.embedding_size,
                                hidden_size=self.hidden_size,
                                batch_first=True)

        self.fc = torch.nn.Linear(self.hidden_size,num_class)

    def forward(self,input):

        hidden = torch.zeros(self.num_layers,self.batch_size,self.hidden_size)
        input = self.embedding(input)        #(batch_size,input_size,embedding_size)
        out,_ = self.rnn(input,hidden)
        out = self.fc(out)
        return out.view(-1,num_class)


net = model(input_size,hidden_size,batch_size,embedding_size)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(),lr=0.05)

for epoch in range(30):
    optimizer.zero_grad()
    out = net(inputs)
    loss = criterion(out,labels)
    loss.backward()
    optimizer.step()

    _,idx = out.max(dim=1)
    idx = idx.data.numpy()
    print('predict:',''.join([idx2char[x] for x in idx]),end=' ')
    print(f'eopch:[{epoch+1}/15],loss={loss.item()}')


# print(inputs.shape)
# print(net.forward(inputs).shape)