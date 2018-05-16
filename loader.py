import dlc_bci as bci
import torch
import numpy as np
from scipy.signal import savgol_filter


"""
Load data
---------
Input paramters:
- data_aug: if True, creates an augmented dataset from the 1000 Hz dataset by
    down-sampling at frequency 100 Hz. In this case the number of samples in the
    training set is multiplied by a factor 10, i.e. 3160 samples of size 28x50.
    The test dataset is never augmented
- data_long: if True, uses the 1000 Hz dataset instead of the down-sampled one,
    i.e., the training dataset will have size 316x28x500
- filtered: if True, adds a smoothing to all datasets (training, test and validation).
    There are two options of filtering: a convolutional mask (option = 0 ) or the
    Savitzky Golay filter (option = 1). The "option" choice is hard-coded. The
    filtering does not change the dimension of the dataset
- filtered_load: if True, loads a pre-filtered dataset.
    This file has to exist in the data_bci folder
- cv_perc: it corresponds to the percentage of training samples to be put in the
    validation set

Output parameters:
- train_input
- train_target
- test_input
- test_target
- validation_input
- validation_target
"""
def load_data(data_aug=False, data_long=False, filtered=False, \
              filtered_load=False, cv_perc=0.2):

    if data_aug and data_long:
        raise RuntimeError("Can not perform data augmentation on 1000 Hz")

    if not filtered and filtered_load:
        print("Warning: Can not load the filtered version of the training input if filtered = False")

    # one_khz = True loads the 1000 Hz datasets for train. This is needed for both
    # when augmenting the dataset or when using the high resolution one
    if data_aug or data_long:
        one_khz = True
    else:
        one_khz = False

    # Use the provided bci loader
    train_input, train_target = bci.load(root='./data_bci',one_khz=one_khz)
    test_input, test_target = bci.load(root='./data_bci',train=False,one_khz=data_long)

    # Compute sizes
    train_size = train_input.size(0)
    test_size = test_input.size(0)
    validation_size = round(cv_perc * train_size)

    # Create a random shuffling in the ordering of the rows (i.e. number of
    # samples) of the train dataset
    indices = torch.randperm(train_size)

    # Select the indices to be used for train and validation sets
    indices_train = indices[0:train_size-validation_size]
    indices_validation = indices[train_size-validation_size+1:]

    if data_aug:
        train_input_list = []
        train_target_list = []
        # Down-sample the 1000Hz signal at frequency 100Hz
        # Initial train_input size: 316 x 28 x 500
        # Final train_input size: 3160 x 28 x 50

        # Create lists of down-sampled signals
        for i in range(10):
            train_input_list.append(train_input[indices_train,:,i::10])
            train_target_list.append(train_target[indices_train])

        # Remark: the validation set is not augmented. To create the validation set,
        # we down-sample just once the 1000Hz signals starting from the 7th timestep,
        # because this is the first timestep in the provided down-sampled set
        validation_input = train_input[indices_validation,:,6::10]
        validation_target = train_target[indices_validation]

        # Create tensors starting from the lists
        train_input = torch.cat(train_input_list,dim=0)
        train_target = torch.cat(train_target_list,dim=0)

    else:
        if (percentage != 0):
            # Select the samples corresponding to validation indices
            validation_input = train_input[indices_validation,:,:]
            validation_target = train_target[indices_validation]
        # Handle the case in which validation is not required
        else:
            validation_input = torch.LongTensor([])
            validation_target = torch.LongTensor([])

        # Select the samples corresponding to train indices
        train_input = train_input[indices_train,:,:]
        train_target = train_target[indices_train]

    channels = train_input.size(1)
    time_steps = train_input.size(2)

    if filtered:

        # Select one of the available masks
        #option = 0  # mask
        option = 1 # savgol filter

        if option == 1:
                mask_size = 5
                mask = np.ones(mask_size)/mask_size
        else:
            window = 7
            order_poly = 3

        #TODO: puoi spiegare quello che fai qui? @Cate
        if filtered_load == 0:

            test_input_smooth_mat = np.zeros((test_samples,channels*time_steps))
            train_input_smooth_mat = np.zeros((train_samples,channels*time_steps))
            validation_input_smooth_mat = np.zeros((validation_samples,channels*time_steps))

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

            for i in range(validation_samples):
                print(i)
                for j in range(channels):
                    if option == 0:
                        validation_input_smooth_mat[i,j*time_steps:(j+1)*time_steps] = np.convolve(validation_input[i,j,:],mask,mode='same')
                    else:
                        validation_input_smooth_mat[i,j*time_steps:(j+1)*time_steps] = savgol_filter(validation_input[i,j,:].numpy(),window,order_poly)
                    validation_input[i,j,:] = torch.tensor(validation_input_smooth_mat[i,j*time_steps:(j+1)*time_steps])

            # Save the text file of smoothed data
            filename_test = 'data_bci/test_smooth'+str(option)
            filename_train = 'data_bci/train_smooth'+str(option)
            filename_validation = 'data_bci/validation_smooth'+str(option)

            if option == 0:
                filename_test=filename_test+'_mask'+str(mask_size)
                filename_train=filename_train+'_mask'+str(mask_size)
                filename_validation=filename_validation+'_mask'+str(mask_size)
            else:
                filename_test=filename_test+'_w'+str(window)+'_or'+str(order_poly)
                filename_train=filename_train+'_w'+str(window)+'_or'+str(order_poly)
                filename_validation=filename_validation+'_w'+str(window)+'_or'+str(order_poly)

            torch.save(test_input,filename_test+'.pt')
            torch.save(train_input,filename_train+'.pt')
            torch.save(validation_input,filename_validation+'.pt')


            np.savetxt(filename_test+'.txt', test_input_smooth_mat)
            np.savetxt(filename_train+'.txt', train_input_smooth_mat)
            np.savetxt(filename_validation+'.txt', validation_input_smooth_mat)

        else: # load only

            # Load filtered datasets from file
            filename_test = 'data_bci/train_smooth'+str(option)
            filename_train = 'data_bci/train_smooth'+str(option)
            filename_validation = 'data_bci/validation_smooth'+str(option)

            if option == 0:
                filename_test=filename_test+'_mask'+str(mask_size)
                filename_train=filename_train+'_mask'+str(mask_size)
                filename_validation=filename_validation+'_mask'+str(mask_size)
            else:
                filename_test=filename_test+'_w'+str(window)+'_or'+str(order_poly)
                filename_train=filename_train+'_w'+str(window)+'_or'+str(order_poly)
                filename_validation=filename_validation+'_w'+str(window)+'_or'+str(order_poly)

            torch.load(filename_test+'.pt')
            torch.load(filename_train+'.pt')
            torch.load(filename_validation+'.pt')

    # Normalize the data sets wrt the mean value of each channel of the
    # train dataset
    for i in range(0,channels):
        me_train = torch.mean(train_input[:,i,:])
        std_train = torch.std(train_input[:,i,:])
        train_input[:,i,:] = (train_input[:,i,:] - me_train)/std_train
        test_input[:,i,:] = (test_input[:,i,:] - me_train)/std_train
        validation_input[:,i,:] = (validation_input[:,i,:] - me_train)/std_train

    return train_input, train_target, test_input, test_target, validation_input, validation_target
