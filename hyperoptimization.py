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

import copy

import utilities as util

torch.manual_seed(np.random.randint(0,100000))

########
# Main #
########

# Load data
data_aug      = False
data_long     = False
filtered      = False
filtered_load = False

cv_perc = 0.1
train_input, train_target,test_input, test_target, validation_input, validation_target \
    = loader.load_data(data_aug=data_aug,data_long=data_long,filtered=filtered,\
    filtered_load=filtered_load,cv_perc=cv_perc)

# Compute number of samples in the training and test dataset.
train_size = train_target.size(0)
test_size = test_input.size(0)
validation_size = validation_input.size(0)

while 1:
    # random sampling of hyperparameters
    p = ParametersSampler()
    outputManager = OutputManager()

    n_epochs = 200

    model_list = []
    model_list_initial = []

    outputs = []

    best_test_error = 100
    index_best = 0

    for n_runs in range(1):
        # Definition of model and loss function choice
        model = models.M4_dropout(p.getParameter('size_hidden_layer'), \
                                  p.getParameter('size_kernel'), \
                                  p.getParameter('size_conv1'),\
                                  p.getParameter('size_conv2'),
                                  p.getParameter('dropout'))

        criterion = nn.CrossEntropyLoss()

        # Random initialization of weights
        model.apply(util.init_weights)

        # append current model to list
        model_list.append(model)
        model_list_initial.append(copy.deepcopy(model))

        # Train model using our best paramters
        output = util.train_model(model, criterion, train_input, train_target,  \
             validation_input, validation_target, test_input, test_target, \
             n_epochs, p.getParameter('eta'), p.getParameter('batch_perc'), \
             p.getParameter('l2_parameter'), p.getParameter('scale'))

        outputs.append(output)

        nberrors_train = util.compute_nb_errors(model,train_input, train_target)
        nberrors_test = util.compute_nb_errors(model,test_input, test_target)

        train_error = (nberrors_train/train_size)*100
        test_error = (nberrors_test/test_size)*100

        # keep track of best run for this choice of parameters
        if (test_error < best_test_error):
            index_best = n_runs
            best_test_error = test_error

        train_error_string = "Train error: {0:.2f}%".format(train_error)
        test_error_string = "Test error: {0:.2f}%".format(test_error)
        print(train_error_string)
        print(test_error_string)

    outputManager.write(p,outputs)
    outputManager.writeModel(model_list[index_best],"best_model")
    outputManager.writeModel(model_list_initial[index_best],"best_model_initial")
