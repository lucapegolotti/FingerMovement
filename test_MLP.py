import torch
import loader
import numpy as np
from torch.autograd import Variable
from torch import nn
from torch.nn import functional as F

torch.manual_seed(np.random.randint(0,100000))

train_input, train_target, test_input, test_target = loader.load_data()

train_target = train_target.type(torch.FloatTensor)
test_target = test_target.type(torch.FloatTensor)

train_input = train_input.view(316,1400).squeeze(1)
test_input = test_input.view(100,1400).squeeze(1)

class Net(nn.Module):
    def __init__(self, hidden):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(1400, hidden)
        self.fc2 = nn.Linear(hidden, 1)

    def forward(self, x):
        x = self.fc1(x)
        # x = F.tanh(self.fc2(x))
        x = F.sigmoid(self.fc2(x))
        return x

def train_model(model, train_input, train_target, mini_batch_size):
    for e in range(0, 4000):
        sum_loss = 0
        # We do this with mini-batches
        for b in range(0, train_input.size(0), mini_batch_size):
            output = model.forward(train_input.narrow(0, b, mini_batch_size))
            loss = criterion(output, train_target.narrow(0, b, mini_batch_size))
            sum_loss = sum_loss + loss.data[0]
            model.zero_grad()
            loss.backward()
            for p in model.parameters():
                p.data.sub_(eta * p.grad.data)
        print(e, sum_loss)

def compute_nb_errors(model, input, target, mini_batch_size):
    y = model.forward(input).round_().squeeze_()
    nberrors = torch.norm(y-target,0)
    return nberrors.data[0]

train_input, train_target = Variable(train_input), Variable(train_target)
test_input, test_target = Variable(test_input), Variable(test_target)

# hiddens = [10, 50, 200, 500]
hiddens = [50]
for i in hiddens:
    print("Hidden layer's dimension = {0:d}".format(i))
    model, criterion = Net(i), nn.MSELoss()
    # model, criterion = Net(i), nn.NLLLoss()
    eta, mini_batch_size = 1e-1, 79

    train_model(model, train_input, train_target, mini_batch_size)
    nberrors_train = compute_nb_errors(model,train_input, train_target, mini_batch_size)
    nberrors_test = compute_nb_errors(model,test_input, test_target, mini_batch_size)

    print("Train error: {0:.2f}%".format((nberrors_train/1000)*100))
    print("Test error: {0:.2f}%".format((nberrors_test/1000)*100))
