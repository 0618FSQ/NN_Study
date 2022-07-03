import torch


batch_size = 1
seq_len = 3
input_size = 4
hidden_size = 2
num_layers = 1

# #创建RNN层方法一：
# cell = torch.nn.RNNCell(input_size=128,hidden_size=64)
# #调用
# hidden = cell(input,hidden) #input of shape:(batchsize,input_size)    , hidden of shape:(batchsize,hidden_size)
#
# cell = torch.nn.RNNCell(input_size=input_size,hidden_size=hidden_size)
#
# dataset = torch.randn(seq_len,batch_size,input_size)
# hidden = torch.zeros(batch_size,hidden_size)
#
# for idx,inputs in enumerate(dataset):
#     print('='*20,idx,'='*20)
#     print('inputs size:',inputs.shape)
#     hidden = cell(inputs,hidden)
#
#     print('hidden size:',hidden.shape)
#     print(hidden)

# #创建RNN层方法二：
# cell = torch.nn.RNN(input_size=input_size,hidden_size=hidden_size,num_layers=num_layers)
# out,hidden = cell(input,hidden)  #input of shape (seq_size,batch_size,input_size),  hidden of shape (num_layers,batch_size,hidden_size)

cell = torch.nn.RNN(input_size=input_size,hidden_size=hidden_size,num_layers=num_layers)

input = torch.randn(seq_len,batch_size,input_size)
hidden = torch.randn(num_layers,batch_size,hidden_size)

out,hidden = cell(input,hidden)

print('out shape:',out.shape)
print('out:',out)
print('hidden shape:',hidden.shape)
print('hidden:',hidden)
