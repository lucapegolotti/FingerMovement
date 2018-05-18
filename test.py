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

########
# Main #
########

# if load_best_initial_condition = 1, we take the initial condition of the best
# model we obtain to start the train. Otherwise, we choose random weights for the
# network
load_best_initial_condition = 0
name_best = "best_model/best_model_initial.pt"
# many_run = 0 means that this code with fixed choice of parameters will be run
# once. If instead many_run = 1 the code will be run N_times = 100 times and the
# outputs are saved in a subfolder of folder "output_many_run"
many_run = 0
if many_run :
    N_times = 100
else:
    N_times = 1

# parameters correpond to best choice of parameters for method M4 as described in
# the report. The description of method M4 can be found in models.py
# parameters = {'batch_perc'        : 0.42623900430271944,
#               'eta'               : 0.0002592079785007456,
#               'dropout'           : 0.47626453296997534,
#               'size_conv1'        : 30,
#               'size_conv2'        : 17,
#               'size_kernel'       : 9,
#               'size_hidden_layer' : 138,
#               'l2_parameter'      : 0.00033274567029747243,
#               'gamma'             : 0.8450533716557838
# }

parameters = {'batch_perc'        : 0.19052629062090362,
              'eta'               : 0.001,
              'dropout'           : 0.35,
              'size_conv1'        : 18,
              'size_conv2'        : 12,
              'size_kernel'       : 11,
              'size_hidden_layer' : 122,
              'l2_parameter'      : 0.001,
              'gamma'             : 0.9
}

if many_run:
    outputManager = OutputManager()

# Load data
data_aug      = False
data_long     = False
filtered      = False
filtered_load = False
# No cross-validation dataset is used here. cv_perc = 0.2 was used while training the hyperparameters in hyperoptimization.py
cv_perc = 0.0
train_input, train_target,test_input, test_target, validation_input, validation_target \
    = loader.load_data(data_aug=data_aug,data_long=data_long,\
    filtered=filtered,filtered_load=filtered_load,cv_perc=cv_perc)

# Compute number of samples in the training and test dataset.
train_size = train_target.size(0)
test_size = test_input.size(0)
validation_size = validation_input.size(0)

n_epochs = 200

for n_runs in range(N_times):
    model = models.M4_dropout(parameters['size_hidden_layer'], \
        parameters['size_kernel'], parameters['size_conv1'],\
        parameters['size_conv2'], parameters['dropout'])
    criterion = nn.CrossEntropyLoss()

    # Definition of model and loss function choice
    if load_best_initial_condition:
        model.load_state_dict(torch.load(name_best))
    else:
        # Random initialization of weights
        model.apply(util.init_weights)

    # Train model using our best paramters
    output = util.train_model(model,criterion, train_input, train_target, \
         validation_input, validation_target, test_input, test_target,\
         n_epochs, parameters['eta'], parameters['batch_perc'], \
         parameters['l2_parameter'], parameters['gamma'])

    nberrors_train = util.compute_nb_errors(model,train_input, train_target)
    nberrors_test = util.compute_nb_errors(model,test_input, test_target)

    train_error = (nberrors_train/train_size)*100
    test_error = (nberrors_test/test_size)*100

    train_error_string = "Train error: {0:.2f}%".format(train_error)
    test_error_string = "Test error: {0:.2f}%".format(test_error)
    print(train_error_string)
    print(test_error_string)

    if many_run:
        outputManager.writeOne(output,n_runs)
