import torch
import loader as loader
from parameters_sampler import ParametersSampler
from output_manager import OutputManager

import numpy as np
from torch.autograd import Variable
from torch import nn
from torch.nn import functional as F

import models as models

import random

torch.manual_seed(np.random.randint(0,100000))

parameters = ParametersSampler()
outputManager = OutputManager()

train_input, train_target, test_input, test_target = loader.load_data(data_aug=True)

train_target = train_target.type(torch.FloatTensor)
test_target = test_target.type(torch.FloatTensor)

train_size = train_target.size(0)
test_size = test_input.size(0)

def compute_nb_errors(model, input, target):
    # this is for dropout
    model.train(False)
    y = model.forward(input)
    indicesy = np.argmax(y.data,1).float()

    nberrors = np.linalg.norm(indicesy - target.data,0)
    model.train(True)
    return nberrors

#def train_model(model, train_input, train_target, validation_input, validation_target, eta, mini_batch_size):
def train_model(model, train_input, train_target, validation_input, validation_target, test_input, test_target, eta, mini_batch_size, l2_parameter):
    initial_mini_batch_size = mini_batch_size

    train_size = train_input.size(0)
    test_size = test_input.size(0)
    validation_size = validation_input.size()

    # n_epochs = parameters.getParameter('epochs')
    n_epochs = 1000

    # optimizer = torch.optim.SGD(model.parameters(), lr = eta)
    optimizer = torch.optim.Adam(model.parameters(), lr = eta, weight_decay=l2_parameter)
    scheduler = adaptive_time_step(optimizer)

    for e in range(0, n_epochs):
        mini_batch_size = initial_mini_batch_size
        sum_loss = 0
        # We do this with mini-batches
        for b in range(0, train_input.size(0), mini_batch_size):

            mini_batch_size = min(mini_batch_size, train_input.size(0) - b)
            output = model.forward(train_input.narrow(0, b, mini_batch_size))
            train_target_narrowed = train_target.narrow(0, b, mini_batch_size).long()

            loss = criterion(output, train_target_narrowed)

            scheduler.step()

            sum_loss = sum_loss + loss.data[0]
            model.zero_grad()
            loss.backward()
            optimizer.step()
            #for p in model.parameters():
            #    p.data.sub_(eta * p.grad.data)

        train_error = compute_nb_errors(model,train_input, train_target)
        test_error = compute_nb_errors(model,test_input, test_target)
        print("Epoch = {0:d}".format(e))
        print("Loss function = {0:.8f}".format(sum_loss))
        print("Train error: {0:.2f}%".format((train_error/train_size)*100))
        print("Test error: {0:.2f}%".format((test_error/test_size)*100))
        if validation_size[0] is not 0:
            validation_error = compute_nb_errors(model,validation_input, validation_target)
            print("Validation error: {0:.2f}%".format((validation_error/validation_size[0])*100))

def l2_regularization(parameters,penalty_parameter):
    # See Book (pg.116,Chapter 5)
    penalty_term = 0
    for p in parameters:
       penalty_term += penalty_parameter*p.pow(2).sum()
    return penalty_term

def l1_regularization(parameters,penalty_parameter):
    # See Book (pg.116,Chapter 5)
    penalty_term = 0
    for p in parameters:
       penalty_term += penalty_parameter*abs(p).sum()
    return penalty_term

def init_weights(layer):
    if type(layer) == nn.Linear:
        torch.nn.init.xavier_uniform(layer.weight)

def adaptive_time_step(optimizer):
    # See http://pytorch.org/docs/master/optim.html#how-to-adjust-learning-rate
    # Section "How to adjust Learning Rate"

    # Lambda LR
    # Step LR
    step_size = 30
    gamma = 0.95
    last_epoch = -1
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size, gamma, last_epoch)

    # Exponential LR
    #gamma = 0.99
    #last_epoch = -1
    #scheduler =  torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma, last_epoch)

    # ReduceLR on Plateau
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')

    return scheduler

def create_validation(train_input, train_output, percentage):
    samples = train_input.size(0)
    validation_size = round(percentage * samples)

    indices = torch.LongTensor(np.random.choice(samples, samples))

    if (percentage != 0):
        validation_input = train_input[indices[0:validation_size],:,:]
        validation_output = train_output[indices[0:validation_size]]
    else:
        validation_input = torch.LongTensor([]);
        validation_output = torch.LongTensor([]);

    train_input = train_input[indices[validation_size+1:samples],:,:]
    train_output = train_output[indices[validation_size+1:samples]]

    return train_input, train_output, validation_input, validation_output

# sample parameters

parameters.showMe()

# save value of parameters
batch_size = parameters.getParameter("batch_size")
eta = parameters.getParameter("eta")
dropout = parameters.getParameter("dropout")
size_conv1 = parameters.getParameter("size_conv1")
size_conv2 = parameters.getParameter("size_conv2")
size_kernel = parameters.getParameter("size_kernel")
size_hidden_layer = parameters.getParameter("size_hidden_layer")
l2_parameter = parameters.getParameter("l2_parameter")

for run in range(5):
    train_input, train_target = Variable(train_input), Variable(train_target)
    test_input, test_target = Variable(test_input), Variable(test_target)

    train_input, train_target, validation_input, validation_output = create_validation(train_input, train_target, 0)

    model, criterion = models.ShallowConvNetPredictor(size_hidden_layer,size_kernel,size_conv1,size_conv2), nn.CrossEntropyLoss()
    model.apply(init_weights)

    train_model(model, train_input, train_target, validation_input, validation_output, test_input, test_target, eta, batch_size, l2_parameter)

    nberrors_train = compute_nb_errors(model,train_input, train_target)
    nberrors_test = compute_nb_errors(model,test_input, test_target)

    train_error = (nberrors_train/train_size)*100
    test_error = (nberrors_test/test_size)*100

    train_error_string = "Train error: {0:.2f}%".format(train_error)
    test_error_string = "Test error: {0:.2f}%".format(test_error)
    print(train_error_string)
    print(test_error_string)

    # outputManager.write(train_error_string,test_error_string,parameters)
