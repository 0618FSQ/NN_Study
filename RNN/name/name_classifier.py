import torch
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.data import Dataset,DataLoader
import gzip
import csv
from tqdm import tqdm
import matplotlib.pyplot as plt

BATCH_SIZE = 256
N_LAYERS = 2
N_EPOCH = 100
N_CHARS = 128
HIDDEN_SIZE = 100
USE_GPU = True

class NameDataset(Dataset):
    def __init__(self,is_train=True):
        file_path = './data/names_train.csv.gz' if is_train else './data/names_test.csv.gz'
        with gzip.open(file_path,'rt') as f:
            reader = csv.reader(f)
            rows = list(reader)
        self.names = [row[0] for row in rows]
        self.countries = [row[1] for row in rows]
        self.len = len(self.names)
        self.country_list = list(sorted(set(self.countries)))
        self.num_country = len(self.country_list)
        self.country_dict = self.getCountryDict()

    def __getitem__(self, item):
        return self.names[item],self.country_dict[self.countries[item]]

    def __len__(self):
        return self.len

    def getCountryDict(self):
        Dict = {}
        for idx,country in enumerate(self.country_list,0):
            Dict[country] = idx
        return Dict

    def getCountryNum(self):
        return self.num_country

train_set = NameDataset()
train_loader = DataLoader(train_set,BATCH_SIZE,shuffle=True)
test_set = NameDataset(is_train=False)
test_loader = DataLoader(test_set,BATCH_SIZE,shuffle=False)

N_COUNTRY = train_set.getCountryNum()

class RNNClassifier(torch.nn.Module):
    def __init__(self,input_size,hidden_size,output_size,n_layers=1,bidirectional=True):
        super(RNNClassifier,self).__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.n_directional = 2 if bidirectional else 1
        self.n_layers = n_layers

        self.embedding = torch.nn.Embedding(input_size,hidden_size)   #input_size 其实是embedding字典长度
        self.GRU = torch.nn.GRU(input_size=hidden_size,
                                hidden_size=hidden_size,
                                num_layers=n_layers,
                                bidirectional=bidirectional)
        self.fc = torch.nn.Linear(hidden_size*self.n_directional,output_size)

    def _init_hidden(self,batch_size):
        hidden = torch.zeros(self.n_layers*self.n_directional,batch_size,self.hidden_size)
        return hidden

    def make_tensor(self,names,countries):
        #将名字字符串用数字表达
        sequences_and_lengths = [self.name2list(name) for name in names]
        #姓名序列
        name_seqences = [x[0] for x in sequences_and_lengths]
        #序列长度
        seq_len = torch.LongTensor([x[1] for x in sequences_and_lengths])
        #标签
        countries = countries.long()

        #padding
        seq_tensor = torch.zeros(len(name_seqences),seq_len.max()).long()  #Batch_size×Seq_len
        # for idx,(seq,seq_len) in enumerate(zip(name_seqences,seq_len)):
        for idx,(seq,seq_lenth) in enumerate(sequences_and_lengths):   #for循环里的变量也会赋值。。。若将seq_lenth改为seq_len则seq_len变为‘int’
            seq_tensor[idx,:seq_lenth] = torch.LongTensor(seq)

        #排序
        seq_len,perm_idx = seq_len.sort(dim=0,descending=True)
        seq_tensor = seq_tensor[perm_idx]
        countries = countries[perm_idx]

        return seq_tensor,seq_len,countries

    def forward(self,input,seq_len):
        #input Shape:Batch_size × Seq_len
        input = input.t()
        batch_size = input.size(1)

        hidden = self._init_hidden(batch_size)

        embedding = self.embedding(input)    #Seq_len×Batch_size×embedding_size

        gru_input = pack_padded_sequence(embedding,seq_len)

        out,hidden = self.GRU(gru_input,hidden)
        if self.n_directional == 2:
            hidden_cat = torch.cat([hidden[-1],hidden[-2]],dim=1)
        else:
            hidden_cat = hidden[-1]
        fc_output = self.fc(hidden_cat)

        return fc_output

    def name2list(self,name):
        arr = [ord(c) for c in name]
        return arr,len(arr)

def trainModel():
    total_loss = 0
    for idx,(names,countries) in enumerate(train_loader,1):
        inputs,seq_len,labels = classifier.make_tensor(names,countries)
        outputs = classifier(inputs,seq_len)
        loss = criterion(outputs,labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    print(total_loss)
    return total_loss

def testModel():
    correct = 0
    total = len(test_set)
    with torch.no_grad():
        for idx,(names,countries) in enumerate(test_loader,1):
            inputs,seq_len,labels = classifier.make_tensor(names,countries)
            outputs = classifier(inputs,seq_len)
            _,pred = torch.max(outputs,dim=1)
            correct += (pred==labels).sum().item()
        print('correctRate:',correct/total)
    return correct/total


if __name__ == '__main__':
    classifier = RNNClassifier(N_CHARS, HIDDEN_SIZE, N_COUNTRY)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(classifier.parameters(), lr=0.001)
    acc_list = []
    loss_list = []
    for epoch in tqdm(range(20)):
        print('*'*20,'trainning epoch:',epoch,'*'*20)
        loss = trainModel()
        loss_list.append(loss)
        acc = testModel()
        acc_list.append(acc*100)

print(loss_list)
plt.plot(loss_list)
plt.plot(acc_list)
plt.show()