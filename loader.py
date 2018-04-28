import dlc_bci as bci
import torch
import numpy

def load_data():
    train_input, train_target = bci.load(root='./data_bci')

    print(str(type(train_input)),train_input.size())
    print(str(type(train_target)),train_target.size())

    test_input, test_target = bci.load(root='./data_bci',train=False)

    print(str(type(test_input)),test_input.size())
    print(str(type(test_target)),test_target.size())

    train_samples = train_input.size(0)
    test_samples = test_input.size(0)
    channels = train_input.size(1)

    for i in range(0,channels):
        me = torch.mean(train_input[:,i,:])
        std = torch.std(train_input[:,i,:])
        train_input[:,i,:] = (train_input[:,i,:] - me)/std;
        test_input[:,i,:] = (test_input[:,i,:] - me)/std;

    return train_input, train_target, test_input, test_target
