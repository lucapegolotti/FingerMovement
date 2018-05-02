import torch
import loader
from parameters_sampler import ParametersSampler
from output_manager import OutputManager

import numpy as np
from torch.autograd import Variable 
from torch import nn
from torch.nn import functional as F

import random


convf = nn.Conv1d(28, 224, kernel_size=6, groups=28, bias=False)
print(convf.weight.size())
print(convf.weight[1:])

batch_size = 30
test = Variable(torch.randn(batch_size,28,50))

c = convf(test)
print(c.size())
print(c)

