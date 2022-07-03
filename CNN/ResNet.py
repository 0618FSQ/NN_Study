import torch


class Residual(torch.nn.Module):
    def __init__(self,input_size,num_output,strides=1,use_1conv=False):
        super(Residual, self).__init__()
        self.relu = torch.nn.ReLU()
        self.conv1 = torch.nn.Conv2d(input_size,num_output,
                                     kernel_size=(3,3),
                                     padding=1,
                                     stride=strides)
        self.conv2 = torch.nn.Conv2d(num_output,num_output,kernel_size=(3,3),padding=1)
        self.bn1 = torch.nn.BatchNorm2d(num_output)
        self.bn2 = torch.nn.BatchNorm2d(num_output)

        if use_1conv:
            self.conv3 = torch.nn.Conv2d(input_size,num_output,kernel_size=1,stride=strides)
        else:
            self.conv3 = None
    def forward(self,x):
        y = self.relu(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))

        if self.conv3:
            x = self.conv3(x)
        y += x

        return self.relu(y)

b1 = torch.nn.Sequential(torch.nn.Conv2d(1,64,kernel_size=(7,7),padding=3,stride=2),
                         torch.nn.BatchNorm2d(64),
                         torch.nn.ReLU(),
                         torch.nn.MaxPool2d(kernel_size=(3,3),stride=2,padding=1))

def resnet_block(input_channels,num_channels,num_residuals,first_block=False):
    blk = []
    for i in range(num_residuals):
        if i==0 and not first_block:
            blk.append(Residual(input_channels,num_channels,use_1conv=True,strides=2))
        else:
            blk.append(Residual(num_channels,num_channels))
        return blk

b2 = torch.nn.Sequential(*resnet_block(64,64,2,first_block=True))
b3 = torch.nn.Sequential(*resnet_block(64,128,2))
b4 = torch.nn.Sequential(*resnet_block(128,256,2))
b5 = torch.nn.Sequential(*resnet_block(256,512,2))

net = torch.nn.Sequential(b1,b2,b3,b4,b5,torch.nn.AdaptiveAvgPool2d((1,1)),
                          torch.nn.Flatten(),torch.nn.Linear(512,10))

x = torch.rand(1,1,224,224)
for layer in net:
    x = layer(x)
    print(layer.__class__.__name__,'output size:\t',x.shape)
