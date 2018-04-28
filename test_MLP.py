import torch
import loader
from parameters_sampler import ParametersSampler
from output_manager import OutputManager

import numpy as np
from torch.autograd import Variable
from torch import nn
from torch.nn import functional as F

import random

torch.manual_seed(np.random.randint(0,100000))

parameters = ParametersSampler()
parameters.showMe()

outputManager = OutputManager()

train_input, train_target, test_input, test_target = loader.load_data()

train_target = train_target.type(torch.FloatTensor)
test_target = test_target.type(torch.FloatTensor)

train_size = train_target.size(0)
test_size = test_input.size(0)
#
# train_input = train_input.view(316,1400).squeeze(1)
# test_input = test_input.view(100,1400).squeeze(1)

class Net(nn.Module):
    def __init__(self,hidden):
        super(Net, self).__init__()
        self.conv1 = nn.Conv1d(28, 28, kernel_size=5, padding=2)
        self.conv2 = nn.Conv1d(28, 28, kernel_size=3, padding=1)
        self.fc_pos = nn.Linear(1400, 40);
        self.fc_vel = nn.Linear(1400, 40)
        self.fc_acc = nn.Linear(1400, 40)
        self.fc_sum = nn.Linear(40, 28)
        self.fc_final = nn.Linear(28,2)

    def forward(self, x):
        v = F.relu(F.max_pool1d(self.conv1(x), kernel_size=1, stride=1))
        a = F.relu(F.max_pool1d(self.conv2(v), kernel_size=1, stride=1))

        x = F.relu(self.fc_pos(x.view(-1, 1400)))
        v = F.relu(self.fc_vel(v.view(-1, 1400)))
        a = F.relu(self.fc_acc(a.view(-1, 1400)))

        sum = x + v + a
        sum = F.sigmoid(self.fc_sum(sum))

        res = self.fc_final(sum)
        return res

class MC_DCNNNet(nn.Module):
    def __init__(self):
        super(MC_DCNNNet, self).__init__()
        self.conv1 = nn.Conv1d(28, 64, kernel_size=6)
        self.conv2 = nn.Conv1d(64, 64, kernel_size=4)
        self.fc1 = nn.Linear(256, 100)
        self.fc2 = nn.Linear(100, 50)
        self.fc3 = nn.Linear(50, 2)

    def forward(self, x):
        x = F.relu(F.avg_pool1d(self.conv1(x), kernel_size=3, stride=3))
        x = F.relu(F.avg_pool1d(self.conv2(x), kernel_size=3, stride=3))
        x = F.relu(self.fc1(x.view(-1, 256)))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# this does not work well
class MC_DCNNNet_separatechannels(nn.Module):
    def __init__(self):
        super(MC_DCNNNet_separatechannels, self).__init__()

        self.conv1 = []
        self.conv2 = []

        for i in range(0,28):
            self.conv1 = self.conv1 + [nn.Conv1d(1, 5, kernel_size=6)]
            self.conv2 = self.conv2 + [nn.Conv1d(5,1, kernel_size=4)]

        self.fc1 = nn.Linear(112, 100)
        self.fc2 = nn.Linear(100, 50)
        self.fc3 = nn.Linear(50, 2)

    def forward(self, x):
        xsize = x.size()
        y = torch.Tensor(xsize[0],xsize[1],4)
        for i in range(0,28):
            xsample = x[:,i,:].unsqueeze(1)
            xsample = F.relu(F.avg_pool1d(self.conv1[i](xsample), kernel_size=3, stride=3))
            xsample = F.relu(F.avg_pool1d(self.conv2[i](xsample), kernel_size=3, stride=3))
            y[:,i,:] = xsample.view(xsize[0],-1)
        x = F.relu(self.fc1(y.view(-1, 112)))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class Net2(nn.Module):
    def __init__(self,hidden1,hidden2):
        super(Net2, self).__init__()
        self.conv1 = nn.Conv1d(28, 32, kernel_size=3)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3)
        self.fc1 = nn.Linear(448, 100)
        self.fc3 = nn.Linear(hidden2, 2)

    def forward(self, x):
        x = F.relu(F.max_pool1d(self.conv1(x), kernel_size=3, stride=3))
        x = F.relu(F.max_pool1d(self.conv2(x), kernel_size=2, stride=2))
        x = F.relu(F.max_pool1d(self.conv3(x), kernel_size=5, stride=5))
        x = F.relu(self.fc1(x.view(-1, 128)))
        x = self.fc2(x)
        x = self.fc3(x)
        return x

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

    n_epochs = parameters.getParameter('epochs')

    optimizer = torch.optim.SGD(model.parameters(), lr = eta)
    scheduler = adaptive_time_step(optimizer)

    penalty_parameter_2 = 0.001
    penalty_parameter_1 = 0.0025

    for e in range(0, n_epochs):
        mini_batch_size = initial_mini_batch_size
        sum_loss = 0
        # We do this with mini-batches
        for b in range(0, train_input.size(0), mini_batch_size):

            mini_batch_size = min(mini_batch_size, train_input.size(0) - b)
            output = model.forward(train_input.narrow(0, b, mini_batch_size))
            train_target_narrowed = train_target.narrow(0, b, mini_batch_size).long()

            loss = criterion(output, train_target_narrowed)
            l2_penalty = l2_regularization(model.parameters(),penalty_parameter_2)
            l1_penalty = l1_regularization(model.parameters(),penalty_parameter_1)

            loss += l2_penalty+l1_penalty

            scheduler.step(loss)

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
    #step_size = 50
    #gamma = 0.95
    #last_epoch = -1
    #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size, gamma, last_epoch)

    # Exponential LR
    #gamma = 0.99
    #last_epoch = -1
    #scheduler =  torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma, last_epoch)

    # ReduceLR on Plateau
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')

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

model, criterion = MC_DCNNNet(), nn.CrossEntropyLoss()
model.apply(init_weights)

eta, mini_batch_size = parameters.getParameter('eta'), parameters.getParameter('batch_size')
train_model(model, train_input, train_target, validation_input, validation_output, test_input, test_target, eta, mini_batch_size)

nberrors_train = compute_nb_errors(model,train_input, train_target)
nberrors_test = compute_nb_errors(model,test_input, test_target)

train_error = (nberrors_train/train_size)*100
test_error = (nberrors_test/test_size)*100

train_error_string = "Train error: {0:.2f}%".format(train_error)
test_error_string = "Test error: {0:.2f}%".format(test_error)
print(train_error_string)
print(test_error_string)

outputManager.write(train_error_string,test_error_string,parameters)
