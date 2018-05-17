import torch
from torch import nn

import numpy as np
import random

"""
Compute number of errors on given datset wrt its target
"""
def compute_nb_errors(model, input, target):
    # To deactivate dropout:
    model.train(False)

    y = model.forward(input)
    indicesy = np.argmax(y.data,1).float()

    nberrors = np.linalg.norm(indicesy - target.data.float(),0)

    model.train(True)

    return nberrors

"""
Train model
"""
def train_model(model, criterion, train_input, \
    train_target, validation_input, validation_target, test_input, test_target,\
    n_epochs=1000, eta=0.1, batch_perc=0.3, l2_parameter=1e-3, gamma=0.95):

    train_size = train_input.size(0)
    test_size = test_input.size(0)
    validation_size = validation_input.size(0)

    # Initialize an output_array of 4 or 5 columns if there isn't or there is
    # a validation set, respectively
    if validation_size is not 0:
        output_array = np.zeros(shape=(n_epochs,5))
    else:
        output_array = np.zeros(shape=(n_epochs,4))


    # Effective mini-batch size for this training set
    mini_batch_size = int(train_size*batch_perc)

    # Adam otpimizer to improve weights and biases after each epoch
    optimizer = torch.optim.Adam(model.parameters(), lr = eta, weight_decay = l2_parameter)

    # We adjust the learning rate following a geometric progression with coefficient
    # gamma every 30 epochs
    # See http://pytorch.org/docs/master/optim.html#how-to-adjust-learning-rate
    # Section "How to adjust Learning Rate"
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=gamma, last_epoch=-1)

    for e in range(0, n_epochs):

        # Initialization of loss for SGD with mini-batches after random shuffling
        # of training dataset
        sum_loss = 0

        shuffle_indexes_minibatch = torch.randperm(train_size)
        train_input = train_input[shuffle_indexes_minibatch,:,:]
        train_target  = train_target[shuffle_indexes_minibatch]

        for b in range(0, train_size, mini_batch_size):

            # To avoid errors when train_size/mini_batch_size is not an integer
            # we introduce the variable current_mini_bach_size
            current_mini_batch_size = min(mini_batch_size, train_input.size(0) - b)

            # Forward run
            output = model.forward(train_input.narrow(0, b, current_mini_batch_size))

            # Take the mini-batch of train_target
            train_target_narrowed = train_target.narrow(0, b, current_mini_batch_size).long()

            # Compute the loss
            loss = criterion(output, train_target_narrowed)
            scheduler.step()
            sum_loss = sum_loss + loss.data[0]

            # Backward step
            model.zero_grad()
            loss.backward()
            optimizer.step()


        # Compute number of erorrs on training and test set
        train_error = compute_nb_errors(model,train_input, train_target)
        test_error = compute_nb_errors(model,test_input, test_target)
        # print("Epoch = {0:d}".format(e))
        # print("Loss function = {0:.8f}".format(sum_loss))
        string_to_print = "Epoch: {0:d}".format(e) + \
                          " loss function: {0:.8f}".format(sum_loss) + \
                          " train error: {0:.2f}%".format((train_error/train_size)*100)

        # Save results on output_array to be exported
        output_array[e,0] = e
        output_array[e,1] = sum_loss
        output_array[e,2] = (train_error/train_size)*100
        output_array[e,3] = (test_error/test_size)*100

        # If the cross-validation set is non empty we can also compute the
        # number of errors on this set
        if validation_size is not 0:
            validation_error = compute_nb_errors(model,validation_input, validation_target)
            string_to_print += " validation error: {0:.2f}%".format((validation_error/validation_size)*100)
            output_array[e,4] = (validation_error/validation_size)*100
        print(string_to_print)
    print("===============================================================================")
    return output_array

"""
Random initialization of weights using Xavier uniform
"""
def init_weights(layer):
    if type(layer) == nn.Linear:
        torch.nn.init.xavier_uniform(layer.weight)
