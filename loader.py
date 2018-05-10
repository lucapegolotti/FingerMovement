import dlc_bci as bci
import torch
import numpy as np

def load_data(data_aug=False):
    train_input, train_target = bci.load(root='./data_bci',one_khz=data_aug)

    test_input, test_target = bci.load(root='./data_bci',train=False)

    train_samples = train_input.size(0)
    test_samples = test_input.size(0)
    channels = train_input.size(1)

    mask_size = 1
    mask = np.ones(mask_size)/mask_size

    # for i in range(test_samples):
    #     for j in range(channels):
    #         test_input[i,j,:] = torch.tensor(np.convolve(test_input[i,j,:],mask,mode='same'))
    #
    # for i in range(train_samples):
    #     for j in range(channels):
    #         train_input[i,j,:] = torch.tensor(np.convolve(train_input[i,j,:],mask,mode='same'))

    for i in range(0,channels):
        me_train = torch.mean(train_input[:,i,:])
        std_train = torch.std(train_input[:,i,:])
        train_input[:,i,:] = (train_input[:,i,:] - me_train)/std_train;
        test_input[:,i,:] = (test_input[:,i,:] - me_train)/std_train;

    # for i in range(0,channels):
    #     for j in range(0,train_samples):
    #         min_train = torch.min(train_input[j,i,:])
    #         max_train = torch.max(train_input[j,i,:])
    #         train_input[j,i,:] = (train_input[j,i,:] - min_train) / (max_train - min_train)
    #
    #     for j in range(0,test_samples):
    #         min_test = torch.min(test_input[j,i,:])
    #         max_test = torch.max(test_input[j,i,:])
    #         test_input[j,i,:] = (test_input[j,i,:] - min_test) / (max_test - min_test)

    if data_aug:
        train_input_list = []
        train_target_list = []
        for i in range(10):
            train_input_list.append(train_input[:,:,i::10])
            train_target_list.append(train_target)


        train_input = torch.cat(train_input_list,dim=0)
        train_target = torch.cat(train_target_list,dim=0)

    return train_input, train_target, test_input, test_target
