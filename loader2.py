import dlc_bci as bci
import torch
import numpy as np
import random

def load_data(data_aug=False):
    # train_input, train_target = bci.load(root='./data_bci',one_khz=True)
    train_input, train_target = bci.load(root='./data_bci',train=True,one_khz=True)

    test_input, test_target = bci.load(root='./data_bci',train=True)

    train_samples = train_input.size(0)
    test_samples = test_input.size(0)
    channels = train_input.size(1)

    for i in range(0,channels):
        me_train = torch.mean(train_input[:,i,:])
        std_train = torch.std(train_input[:,i,:])
        train_input[:,i,:] = (train_input[:,i,:] - me_train)/std_train;
        test_input[:,i,:] = (test_input[:,i,:] - me_train)/std_train;

    # if data_aug:
    #     train_input_list = []
    #     train_target_list = []
    #     for i in range(10):
    #         train_input_list.append(train_input[:,:,i::10])
    #         train_target_list.append(train_target)
    #
    #     train_input = torch.cat(train_input_list,dim=0)
    #     train_target = torch.cat(train_target_list,dim=0)

    if data_aug:
        train_input_list = []
        train_target_list = []
        for i in range(10):
            train_input_list.append(train_input[0:237,:,i::10])
            train_target_list.append(train_target[0:237])

        train_input = torch.cat(train_input_list,dim=0)
        train_target = torch.cat(train_target_list,dim=0)

    # list_input = []
    # list_target = []
    #
    # list_input.append(train_input)
    # list_target.append(train_target)
    # # list_input.append(test_input)
    # # list_target.append(test_target)
    #
    # train_input_full = torch.cat(list_input,dim=0)
    # train_target_full = torch.cat(list_target,dim=0)
    #
    #
    # indices = torch.randperm(train_input_full.size(0))
    #
    # train_input = train_input_full[indices[0:237],:,:]
    # train_target = train_target_full[indices[0:237]]
    # test_input = train_input_full[indices[237:316],:,:]
    # test_target = train_target_full[indices[237:316]]


    return train_input, train_target, test_input, test_target
