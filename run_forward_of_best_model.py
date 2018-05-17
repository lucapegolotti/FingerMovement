import torch
from torch.autograd import Variable
from torch import nn
from torch.nn import functional as F

import loader as loader
import models as models

from parameters_sampler import ParametersSampler
from output_manager import OutputManager

import numpy as np
import random

import utilities as util

torch.manual_seed(np.random.randint(0,100000))

name_best = "best_model/best_model.pt"

parameters = {'dropout'           : 0.47626453296997534,
              'size_conv1'        : 18,
              'size_conv2'        : 12,
              'size_kernel'       : 11,
              'size_hidden_layer' : 122,
}

data_aug      = False
data_long     = False
filtered      = False
filtered_load = False
# Load the datasat
train_input, train_target,test_input, test_target, validation_input, validation_target \
    = loader.load_data(data_aug=data_aug,data_long=data_long,\
    filtered=filtered,filtered_load=filtered_load,cv_perc=0.0)

# Compute number of samples in the training and test dataset.
train_size = train_target.size(0)
test_size = test_input.size(0)
validation_size = validation_input.size(0)

model = models.M4_dropout(parameters['size_hidden_layer'], \
    parameters['size_kernel'], parameters['size_conv1'],\
    parameters['size_conv2'], parameters['dropout'])
criterion = nn.CrossEntropyLoss()

# Load best model from datafile
model.load_state_dict(torch.load(name_best))

nberrors_train = util.compute_nb_errors(model,train_input, train_target)
nberrors_test = util.compute_nb_errors(model,test_input, test_target)

train_error = (nberrors_train/train_size)*100
test_error = (nberrors_test/test_size)*100

train_error_string = "Train error: {0:.2f}%".format(train_error)
test_error_string = "Test error: {0:.2f}%".format(test_error)
print(train_error_string)
print(test_error_string)
