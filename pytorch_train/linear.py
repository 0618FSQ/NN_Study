import random
import torch
from d2l import torch as d2l

def synthetic_data(w,b,num_examples):
    """生成 y = xw + b +噪声"""
    x = torch.normal(0,1,(num_examples,len(w)))  #均值为0 方差为1 形状为（，）
    y = torch.matmul(x,w) + b    #等同于矩阵乘法
    y += torch.normal(0,0.01,y.shape)
    return x,y.reshape((-1,1))

true_w = torch.tensor([2,-3.4])
true_b = 4.2

features,labels = synthetic_data(true_w,true_b,1000)

print(len(true_w))