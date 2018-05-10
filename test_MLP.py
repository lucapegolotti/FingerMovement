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

# parameters = ParametersSampler()
# parameters.showMe()

outputManager = OutputManager()

train_input, train_target, test_input, test_target = loader.load_data(data_aug=True,data_long=False,filtered=True,filtered_load=True)


train_target = train_target.type(torch.FloatTensor)
test_target = test_target.type(torch.FloatTensor)

train_size = train_target.size(0)
test_size = test_input.size(0)

def compute_nb_errors(model, input, target):
    y = model.forward(input)
    indicesy = np.argmax(y.data,1).float()

    nberrors = np.linalg.norm(indicesy - target.data,0)

    return nberrors

#def train_model(model, train_input, train_target, validation_input, validation_target, eta, mini_batch_size):
def train_model(model, train_input, train_target, validation_input, validation_target, test_input, test_target, eta, mini_batch_size):
    initial_mini_batch_size = mini_batch_size

    train_size = train_input.size(0)
    test_size = test_input.size(0)
    validation_size = validation_input.size()

    # n_epochs = parameters.getParameter('epochs')
    n_epochs = 1000

    # optimizer = torch.optim.SGD(model.parameters(), lr = eta)
    optimizer = torch.optim.Adam(model.parameters(), lr = eta)
    scheduler = adaptive_time_step(optimizer)

    penalty_parameter_2 = 0.001
    penalty_parameter_1 = 0.001

    for e in range(0, n_epochs):
        mini_batch_size = initial_mini_batch_size
        sum_loss = 0
        # We do this with mini-batches
        for b in range(0, train_input.size(0), mini_batch_size):

            mini_batch_size = min(mini_batch_size, train_input.size(0) - b)
            output = model.forward(train_input.narrow(0, b, mini_batch_size))
            train_target_narrowed = train_target.narrow(0, b, mini_batch_size).long()

            loss = criterion(output, train_target_narrowed)
            # l2_penalty = l2_regularization(model.parameters(),penalty_parameter_2)
            # l1_penalty = l1_regularization(model.parameters(),penalty_parameter_1)

            l2_penalty = 1e-10
            l1_penalty = 1e-10

            loss += l2_penalty+l1_penalty

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

train_input, train_target = Variable(train_input), Variable(train_target)
test_input, test_target = Variable(test_input), Variable(test_target)

train_input, train_target, validation_input, validation_output = create_validation(train_input, train_target, 0)

hidden1 = 100
hidden2 = 100

model, criterion = models.ShallowConvNetPredictor(), nn.CrossEntropyLoss()
model.apply(init_weights)

eta, mini_batch_size = 0.001, 79
train_model(model, train_input, train_target, validation_input, validation_output, test_input, test_target, eta, mini_batch_size)

nberrors_train = compute_nb_errors(model,train_input, train_target)
nberrors_test = compute_nb_errors(model,test_input, test_target)

train_error = (nberrors_train/train_size)*100
test_error = (nberrors_test/test_size)*100

train_error_string = "Train error: {0:.2f}%".format(train_error)
test_error_string = "Test error: {0:.2f}%".format(test_error)
print(train_error_string)
print(test_error_string)

#outputManager.write(train_error_string,test_error_string,parameters)
