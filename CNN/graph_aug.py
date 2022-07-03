import torch
import torchvision
from d2l import torch as d2l

d2l.set_figsize()
img = d2l.Image.open('./cat1.jpg')
d2l.plt.imshow(img)
d2l.plt.show()

def apply(img,aug,num_rows,num_cols,scale=1.5):