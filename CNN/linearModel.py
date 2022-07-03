import numpy as np
import matplotlib.pyplot as plt

from mpl_toolkits import mplot3d

x_data = np.array([1.0,2.0,3.0]).reshape(3,1)
y_data = np.array([2.0,4.0,6.0]).reshape(3,1)

def forward(x,w,b):
    return np.dot(w,x) + b

def loss(x,w,b,y):
    y_pred = forward(x,w,b).reshape(3,1)
    loss = (y-y_pred)**2
    cost = np.sum(loss)/3

    return y_pred,cost

costs = []
for w in np.arange(0.0,4.1,0.1):
    for b in np.arange(-2.0,2.1,0.1):
        print('w=',w)
        print('b=',b)
        y_pred,cost = loss(x_data,w,b,y_data)
        costs.append(cost)
        print(cost)
        print('-'*30)



ax = plt.axes(projection='3d')
x = np.arange(0.0,4.1,0.1)
y = np.arange(-2,2.1,0.1)
x,y = np.meshgrid(x,y)
x = x.reshape(1,-1)
y = y.reshape(1,-1)
x = np.squeeze(x)
y = np.squeeze(y)
print(x.shape)
print(y.shape)

z = np.array(costs)
print(z.shape)


ax.scatter3D(x, y, z)

plt.show()








