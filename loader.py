import dlc_bci as bci
import torch
import numpy

def load_data(data_aug=False):
    train_input, train_target = bci.load(root='./data_bci',one_khz=data_aug)

    test_input, test_target = bci.load(root='./data_bci',train=False)

    train_samples = train_input.size(0)
    test_samples = test_input.size(0)
    channels = train_input.size(1)

    for i in range(0,channels):
        me_train = torch.mean(train_input[:,i,:])
        std_train = torch.std(train_input[:,i,:])
        train_input[:,i,:] = (train_input[:,i,:] - me_train)/std_train;
        test_input[:,i,:] = (test_input[:,i,:] - me_train)/std_train;

    if data_aug:
        train_input_list = []
        train_target_list = []
        for i in range(10):
            train_input_list.append(train_input[:,:,i::10])
            train_target_list.append(train_target)

        train_input = torch.cat(train_input_list,dim=0)
        train_target = torch.cat(train_target_list,dim=0)

    return train_input, train_target, test_input, test_target
