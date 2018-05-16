import torch
import numpy as np
from torch.autograd import Variable
from torch import nn
from torch.nn import functional as F

class M1(nn.Module): # Linear Predictor
    def __init__(self):
        super(M1, self).__init__()
        self.fc1 = nn.Linear(1400, 2)

    def forward(self, x):
        x = x.view(-1, x.size(1) * x.size(2))
        x = self.fc1(x)
        return x

class M2(nn.Module): # MC_DCNNNet
    def __init__(self):
        super(M2, self).__init__()
        self.conv1 = nn.Conv1d(28, 30, kernel_size=6)
        self.conv2 = nn.Conv1d(30, 60, kernel_size=6)
        self.fc1 = nn.Linear(120, 40)
        self.fc2 = nn.Linear(40, 2)

    def forward(self, x):
        x = F.relu(F.avg_pool1d(self.conv1(x), kernel_size=3))
        x = F.relu(F.avg_pool1d(self.conv2(x), kernel_size=5))
        x = F.relu(self.fc1(x.view(-1, 120)))
        x = self.fc2(x)
        return x

class M3(nn.Module): # M3
    def __init__(self):
        super(M3, self).__init__()
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

class M4(nn.Module): #ShallowConvNetPredictor
    def __init__(self, n_hidden = 125, kernel_size = 11, n_conv_1 = 15, n_conv_2=15):
        super(M4, self).__init__()

        self.conv1 = nn.Conv2d(1, n_conv_1 , kernel_size=(1,kernel_size))
        self.conv2 = nn.Conv2d(n_conv_1, n_conv_2, kernel_size=(28,1))

        self.fc1 = nn.Linear( n_conv_2*(50-kernel_size+1)/2, n_hidden)
        self.fc2 = nn.Linear( n_hidden, 2)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = F.relu(self.conv1(x)) # 316x40x28x46
        x = F.relu(self.conv2(x)) # 316x40x1x46
        x = F.max_pool2d(x, kernel_size = (1,2), stride = (1,2))  # 316x40x1x23
        x = x.view(-1, x.size(1)*x.size(3)) # 316x40x23 = 920
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class M4_dropout(nn.Module): #ShallowConvNetPredictorWithDropout
   def __init__(self, n_hidden = 20, kernel_size = 5, n_conv_1 = 40, n_conv_2=40, p_dropout = 0.5):
       super(M4_dropout, self).__init__()

       self.conv1 = nn.Conv2d(1, n_conv_1 , kernel_size=(1,kernel_size))
       self.conv2 = nn.Conv2d(n_conv_1, n_conv_2, kernel_size=(28,1))

       self.fc1 = nn.Linear( int(n_conv_2*(50-kernel_size+1)/2), n_hidden)
       self.fc2 = nn.Linear( n_hidden, 2)

       self.dropout = nn.Dropout(p=p_dropout)

   def forward(self, x):
       x = x.unsqueeze(1)

       x = F.relu(self.conv1(x)) # 316x40x28x46
       x = F.relu(self.conv2(x)) # 316x40x1x46
       x = F.max_pool2d(x, kernel_size = (1,2), stride = (1,2))  # 316x40x1x23
       x = x.view(-1, x.size(1)*x.size(3)) # 316x40x23 = 920
       x = self.dropout(x)
       x = F.relu(self.fc1(x))
       x = self.dropout(x)
       x = self.fc2(x)
       return x

