import torch
import numpy as np


# A = torch.empty(5,3)
# a = torch.Tensor([[5.1,3.2,4],[3,1,5]])
# print(A.size())
# print(a)
#
# print(a[1,2])
#
# ept = np.array([[1,3,2],[3,5,4],[1,4,8]])
# ept = torch.from_numpy(ept)

# x = torch.randn(3,4,requires_grad=True)
# b = torch.randn(3,4,requires_grad=True)
#
# t = x + b
# y = t.sum()
# print(y)


###保存一个模型
# torch.save(model.state_dict(),'model.pkl')
###读取
# model.load_state_dict(torch.load('model.pkl'))

# x = torch.arange(12,dtype=torch.float32)
# print(x)
# print(x.shape)
# print(x.numel())
# print(x.size())
# x = x.view(3,4)
# # x = x.reshape(3,4)
# print(x)

# a = torch.arange(12).view(3,4)
# b = a.clone()  #重新分配空间
# b[1,1] = 0
# a_sum = a.sum(axis=0,keepdims=True)
# a_consum = a.cumsum(axis=0)   #累加运算符
# print(a_consum)
# print(a)

# torch.dot(a,b)    向量内积
# torch.mv(A,x)     矩阵×向量
# torch.mm(A,B)     矩阵×矩阵
#torch.norm(torch.ones(4,9))  F范数 二阶范数

x = torch.arange(4,dtype=torch.float32)
x.requires_grad = True
y = 2 * torch.dot(x,x)
y.backward()
# print(y)
# print(x.grad == 4*x)

x.grad.zero_()
y = x.sum()
y.backward()
print(x.grad)
