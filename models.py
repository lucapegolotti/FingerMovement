import torch
import numpy as np
from torch.autograd import Variable
from torch import nn
from torch.nn import functional as F

class LinearPredictor(nn.Module):
    def __init__(self):
        super(LinearPredictor, self).__init__()
        self.fc1 = nn.Linear(1400, 2)

    def forward(self, x):
        x = x.view(-1, x.size(1) * x.size(2))
        x = self.fc1(x)
        return x

class LinearPredictor2(nn.Module):
    def __init__(self):
        super(LinearPredictor2, self).__init__()
        self.fc1 = nn.Linear(230, 2)
        self.conv1 = nn.Conv1d(28, 10, kernel_size=5)

    def forward(self, x):
        x = F.relu(F.avg_pool1d(self.conv1(x),kernel_size=2))
        x = x.view(-1, x.size(1) * x.size(2))
        x = self.fc1(x)
        return x

class MC_DCNNNet(nn.Module):
    def __init__(self):
        super(MC_DCNNNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, kernel_size=(1,6))
        self.conv2 = nn.Conv2d(20, 1, kernel_size=(1,4))
        self.fc1 = nn.Linear(112, 2)

    def forward(self, x):
        dim1 = x.size(0)
        dim2 = x.size(1)
        dim3 = x.size(2)
        x = x.unsqueeze(1)
        x = F.relu(F.avg_pool2d(self.conv1(x),kernel_size=(1,3)))
        x = F.relu(F.avg_pool2d(self.conv2(x),kernel_size=(1,3)))
        x = self.fc1(x.view(dim1,-1))
        return x

class MC_DCNNNet2(nn.Module):
    def __init__(self):
        super(MC_DCNNNet2, self).__init__()
        self.conv1 = nn.Conv1d(28, 28, kernel_size=6, groups=28, bias=True)
        self.conv2 = nn.Conv1d(224, 112, kernel_size=4, groups=28, bias=True)
        self.fc1 = nn.Linear(252, 25)
        self.fc2 = nn.Linear(25, 16)
        self.fc4 = nn.Linear(1400, 25)
        self.fc3 = nn.Linear(16,2)
        self.activation2 = nn.functional.sigmoid
        self.activation1 = nn.LeakyReLU(0.001)

    def forward(self, x):
        y = self.activation1(F.avg_pool1d(self.conv1(x),kernel_size=5))
        y = self.activation1(self.fc1(y.view(-1, 252)))
        x = self.activation1(self.fc4(x.view(-1, 1400)))
        x = self.activation1(self.fc2(x+y))
        x = self.fc3(x)
        return x

class ShallowConvNetPredictor(nn.Module):
    def __init__(self,n_conv_1 = 10, n_hidden = 100):
        super(ShallowConvNetPredictor, self).__init__()

        self.conv1 = nn.Conv2d(1, 40 , kernel_size=(1,5))
        self.conv2 = nn.Conv2d(40, 40, kernel_size=(28,1))

        self.fc1 = nn.Linear( 920, n_hidden)
        self.fc2 = nn.Linear( n_hidden, 2)

    def forward(self, x):
        x = x.unsqueeze(1)

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, kernel_size = (1,2), stride = (1,2))
        x = x.view(-1, x.size(1)*x.size(3))
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class ShallowConvNetPredictor_1(nn.Module):
    def __init__(self,n_conv_1 = 10, n_hidden = 100):
        super(ShallowConvNetPredictor_1, self).__init__()

        self.conv1 = nn.Conv2d(1, 40 , kernel_size=(1,5))
        self.conv2 = nn.Conv2d(40, 40, kernel_size=(28,1))

        self.fc1 = nn.Linear( 920, 2)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, kernel_size = (1,2), stride = (1,2))
        x = x.view(-1, x.size(1)*x.size(3))
        x = self.fc1(x)
        return x

class ShallowConvNetPredictor_2(nn.Module):
    def __init__(self,n_conv_1 = 10, n_hidden = 100):
        super(ShallowConvNetPredictor_2, self).__init__()

        self.conv1 = nn.Conv2d(1, 40 , kernel_size=(1,5))
        self.conv2 = nn.Conv2d(40, 40, kernel_size=(28,1))

        self.fc1 = nn.Linear( 920, 2)

        self.dropout = nn.Dropout(p=0.75)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = F.relu(self.conv1(x))
        x = self.dropout(x)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, kernel_size = (1,2), stride = (1,2))
        x = x.view(-1, x.size(1)*x.size(3))
        x = self.fc1(x)
        return x

# here there is something weird (some conv nodes are missing)
# class DeepConvNetPredictor(nn.Module):
#     def __init__(self,n_conv_1 = 10, n_hidden = 100):
#         super(DeepConvNetPredictor, self).__init__()
#
#         self.conv1 = nn.Conv2d(1, 40 , kernel_size=(1,5))
#         self.conv2 = nn.Conv2d(40, 40, kernel_size=(28,1))
#
#         self.fc1 = nn.Linear( 920, n_hidden)
#         self.fc2 = nn.Linear( n_hidden, 2)
#
#     def forward(self, x):
#         x = x.unsqueeze(1)
#
#         x = F.relu(self.conv1(x))
#         x = F.relu(self.conv2(x))
#         x = F.max_pool2d(x, kernel_size = (1,2), stride = (1,2))
#         x = x.view(-1, 1,x.size(1),x.size(3))
#         x = F.relu(self.conv3(x))
#         x = F.relu(self.conv4(x))
#         x = F.relu(self.fc1(x))
#         x = self.fc2(x)
#         return x
