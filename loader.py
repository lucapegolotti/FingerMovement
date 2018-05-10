import dlc_bci as bci
import torch
import numpy as np
from scipy.signal import savgol_filter

def create_validation(train_input, train_target, percentage, data_aug):
    samples = train_input.size(0)
    validation_size = round(percentage * samples)


    # indices = torch.LongTensor(np.random.choice(samples, samples))
    indices = torch.randperm(train_input.size(0))

    indices_train = indices[0:samples-validation_size]
    indices_validation = indices[samples-validation_size+1:]
    if data_aug:
        train_input_list = []
        train_target_list = []
        for i in range(10):
            train_input_list.append(train_input[indices_train,:,i::10])
            train_target_list.append(train_target[indices_train])

        validation_input = train_input[indices_validation,:,6::10]
        validation_target = train_input[indices_validation,:,6::10]

        train_input = torch.cat(train_input_list,dim=0)
        train_target = torch.cat(train_target_list,dim=0)

    else:
        if (percentage != 0):
            validation_input = train_input[indices_validation,:,:]
            validation_target = train_target[indices_validation]
        else:
            validation_input = torch.LongTensor([]);
            validation_target = torch.LongTensor([]);

    train_input = train_input[indices_train,:,:]
    train_target = train_target[indices_train]

    print(train_input.size())
    print(validation_input.size())
    print(train_target.size())
    print(validation_target.size())
    return train_input, train_target, validation_input, validation_target.float()

def load_data(data_aug=False,data_long=False,filtered=False,filtered_load=False):
    if data_aug or data_long:
        one_khz = True
    else:
        one_khz = False
    train_input, train_target = bci.load(root='./data_bci',one_khz=one_khz)

    test_input, test_target = bci.load(root='./data_bci',train=False,one_khz=data_long)

    #print(np.convolve(test_input[0,1,:],np.array([1,1,1,1,1])/5,mode='valid'))


    train_samples = train_input.size(0)
    test_samples = test_input.size(0)
    channels = train_input.size(1)
    time_steps = train_input.size(2)

    train_input, train_target, validation_input, validation_target = create_validation(train_input, train_target, 0.2, data_aug)

    if filtered:

        #option = 0  # mask
        option = 1 # savgol filter

        if option == 0:
                mask_size = 5
                mask = np.ones(mask_size)/mask_size
        else:
            window = 5
            order_poly = 2


        if filtered_load ==0:

            test_input_smooth_mat = np.zeros((test_samples,channels*time_steps))
            train_input_smooth_mat = np.zeros((train_samples,channels*time_steps))

            for i in range(test_samples):
                print(i)
                for j in range(channels):
                    if option == 0:
                        test_input_smooth_mat[i,j*time_steps:(j+1)*time_steps] = np.convolve(test_input[i,j,:],mask,mode='same')
                    else:
                        test_input_smooth_mat[i,j*time_steps:(j+1)*time_steps] = savgol_filter(test_input[i,j,:].numpy(),window,order_poly)
                    test_input[i,j,:] = torch.tensor(test_input_smooth_mat[i,j*time_steps:(j+1)*time_steps])

            for i in range(train_samples):
                print(i)
                for j in range(channels):
                    if option == 0:
                        train_input_smooth_mat[i,j*time_steps:(j+1)*time_steps] = np.convolve(train_input[i,j,:],mask,mode='same')
                    else:
                        train_input_smooth_mat[i,j*time_steps:(j+1)*time_steps] = savgol_filter(train_input[i,j,:].numpy(),window,order_poly)
                    train_input[i,j,:] = torch.tensor(train_input_smooth_mat[i,j*time_steps:(j+1)*time_steps])


            # Save the text file of smoothed data
            filename_test = 'data_bci/test_smooth'+str(option)
            filename_train = 'data_bci/train_smooth'+str(option)

            if option == 0:
                filename_test=filename_test+'_mask'+str(mask_size)
                filename_train=filename_train+'_mask'+str(mask_size)
            else:
                filename_test=filename_test+'_w'+str(window)+'_or'+str(order_poly)
                filename_train=filename_train+'_w'+str(window)+'_or'+str(order_poly)

            torch.save(test_input,filename_test+'.pt')
            torch.save(train_input,filename_train+'.pt')


            np.savetxt(filename_test+'.txt', test_input_smooth_mat)
            np.savetxt(filename_train+'.txt', train_input_smooth_mat)


        else: # load only

            filename_test = 'data_bci/train_smooth'+str(option)
            filename_train = 'data_bci/train_smooth'+str(option)

            if option == 0:
                filename_test=filename_test+'_mask'+str(mask_size)
                filename_train=filename_train+'_mask'+str(mask_size)
            else:
                filename_test=filename_test+'_w'+str(window)+'_or'+str(order_poly)
                filename_train=filename_train+'_w'+str(window)+'_or'+str(order_poly)


            torch.load(filename_test+'.pt')
            torch.load(filename_train+'.pt')


    # NORMALIZE
    # ---------
    for i in range(0,channels):
        me_train = torch.mean(train_input[:,i,:]) # 28 medie
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


    return train_input, train_target, test_input, test_target, validation_input, validation_target
