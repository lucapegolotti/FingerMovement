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

train_size = train_target.size(0)
test_size = test_input.size(0)

train_input = train_input.view(316,1400).squeeze(1)
test_input = test_input.view(100,1400).squeeze(1)

class Net(nn.Module):
    def __init__(self, hidden1, hidden2):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(1400, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc3 = nn.Linear(hidden2, 2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.sigmoid(self.fc3(x))
        return x

def train_model(model, train_input, train_target, mini_batch_size):
    for e in range(0, 1000):
        sum_loss = 0
        # We do this with mini-batches
        for b in range(0, train_input.size(0), mini_batch_size):
            output = model.forward(train_input.narrow(0, b, mini_batch_size))
            train_target_narrowed = train_target.narrow(0, b, mini_batch_size).long()

            loss = criterion(output, train_target_narrowed)
            sum_loss = sum_loss + loss.data[0]
            model.zero_grad()
            loss.backward()
            for p in model.parameters():
                p.data.sub_(eta * p.grad.data)
        print(e, sum_loss)

def compute_nb_errors(model, input, target, mini_batch_size):
    y = model.forward(input)
    indicesy = np.argmax(y.data,1).float()

    print(indicesy - target.data)
    nberrors = np.linalg.norm(indicesy - target.data,0)

    return nberrors

train_input, train_target = Variable(train_input), Variable(train_target)
test_input, test_target = Variable(test_input), Variable(test_target)

hidden1 = 50
hidden2 = 50

# model, criterion = Net(hidden1), nn.MSELoss()
model, criterion = Net(hidden1,hidden2), nn.CrossEntropyLoss()
eta, mini_batch_size = 1e-1, 79

train_model(model, train_input, train_target, mini_batch_size)
nberrors_train = compute_nb_errors(model,train_input, train_target, mini_batch_size)
nberrors_test = compute_nb_errors(model,test_input, test_target, mini_batch_size)

print("Train error: {0:.2f}%".format((nberrors_train/train_size)*100))
print("Test error: {0:.2f}%".format((nberrors_test/test_size)*100))
