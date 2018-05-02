
# coding: utf-8

# In[1]:


get_ipython().magic(u'matplotlib inline')

import matplotlib
import matplotlib.pyplot as plt

import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
os.environ["CUDA_VISIBLE_DEVICES"]="CPU"

import torch
import numpy as np
from torch.autograd import Variable
from torch import nn
from torch.nn import functional as F


import dlc_bci as bci


# In[19]:


train_input , train_target = bci.load(root = './data_bci')
print(str(type(train_input)), train_input.size())
print(str(type(train_target)), train_target.size() )
test_input , test_target = bci.load(root = './data_bci ', train = False)
print(str(type(test_input)), test_input.size())
print(str(type(test_target)), test_target.size())
# normalize data
train_input = train_input - torch.mean(train_input)


mean_train = torch.mean(train_input )
std_train = torch.std(train_input )

#train_input = (train_input - mm)/stdd
#test_input = (test_input - mm)/stdd


# # Visualize data-set

# In[28]:


# each sample consists of 28 EEG channels sampled at 1khz for 0.5s.
for i in range(28):
    plt.plot(np.asarray(train_input[40,i,:]))
print (train_target[40])

plt.figure()
for i in range(28):
    plt.plot(np.asarray(train_input[44,i,:]))
print (train_target[44])


# In[15]:


class LinearPredictor(nn.Module):
    def __init__(self,n_hidden = 200):
        super(LinearPredictor, self).__init__()
        self.fc1 = nn.Linear( train_input.size(1)*train_input.size(2), 2)

    def forward(self, x):
        x = x.view(-1, train_input.size(1)*train_input.size(2))
        x = self.fc1(x)
        return x
    
class ShallowConvNetPredictor(nn.Module):
    def __init__(self,n_conv_1 = 10, n_hidden = 100):
        super(ShallowConvNetPredictor, self).__init__()
        
        self.conv1 = nn.Conv2d(1, 40 , kernel_size=(1,5))
        self.conv2 = nn.Conv2d(40, 40, kernel_size=(28,1))
        
        self.fc1 = nn.Linear( 920, n_hidden)
        self.fc2 = nn.Linear( n_hidden, 2)

    def forward(self, x):
        x = x.view(-1,1, train_input.size(1),train_input.size(2))
        
        #print (x.shape)
        x = F.relu(self.conv1(x))
        #print (x.shape)
        x = F.relu(self.conv2(x))
        #print (x.shape)
        x = F.max_pool2d(x, kernel_size = (1,2), stride = (1,2))
        #print (x.shape)
        x = x.view(-1, x.size(1)*x.size(3))
        #print (x.shape)
        x = F.relu(self.fc1(x))
        #print (x.shape)
        x = self.fc2(x)
        return x
    
class ShallowConvNetPredictor_1(nn.Module):
    def __init__(self,n_conv_1 = 10, n_hidden = 100):
        super(ShallowConvNetPredictor_1, self).__init__()
        
        self.conv1 = nn.Conv2d(1, 40 , kernel_size=(1,5))
        self.conv2 = nn.Conv2d(40, 40, kernel_size=(28,1))
        
        self.fc1 = nn.Linear( 920, 2)

    def forward(self, x):
        x = x.view(-1,1, train_input.size(1),train_input.size(2))
        
        #print (x.shape)
        x = F.relu(self.conv1(x))
        #print (x.shape)
        x = F.relu(self.conv2(x))
        #print (x.shape)
        x = F.max_pool2d(x, kernel_size = (1,2), stride = (1,2))
        #print (x.shape)
        x = x.view(-1, x.size(1)*x.size(3))
        #print (x.shape)
        x = self.fc1(x)
        return x
    
class ShallowConvNetPredictor_2(nn.Module):
    def __init__(self,n_conv_1 = 10, n_hidden = 100):
        super(ShallowConvNetPredictor_2, self).__init__()
        
        self.conv1 = nn.Conv2d(1, 40 , kernel_size=(1,5))
        self.conv2 = nn.Conv2d(40, 40, kernel_size=(28,1))
        
        self.fc1 = nn.Linear( 920, 2)
        
        self.dropout = nn.Dropout(p=0.75)

    def forward(self, x):
        x = x.view(-1,1, train_input.size(1),train_input.size(2))
        
        #print (x.shape)
        x = F.relu(self.conv1(x))
        x = self.dropout(x)
        
        #print (x.shape)
        x = F.relu(self.conv2(x))
        
        
        #print (x.shape)
        x = F.max_pool2d(x, kernel_size = (1,2), stride = (1,2))
        #print (x.shape)
        x = x.view(-1, x.size(1)*x.size(3))
        
        #print (x.shape)
        x = self.fc1(x)
        return x
    


    
class DeepConvNetPredictor(nn.Module):
    def __init__(self,n_conv_1 = 10, n_hidden = 100):
        super(ShallowConvNetPredictor, self).__init__()
        
        self.conv1 = nn.Conv2d(1, 40 , kernel_size=(1,5))
        self.conv2 = nn.Conv2d(40, 40, kernel_size=(28,1))
        self.conv1 = nn.Conv2d(1, 40 , kernel_size=(1,5))
        self.conv2 = nn.Conv2d(40, 40, kernel_size=(28,1))
        
        self.fc1 = nn.Linear( 920, n_hidden)
        self.fc2 = nn.Linear( n_hidden, 2)

    def forward(self, x):
        x = x.view(-1,1, train_input.size(1),train_input.size(2))
        
        #print (x.shape)
        x = F.relu(self.conv1(x))
        #print (x.shape)
        x = F.relu(self.conv2(x))
        #print (x.shape)
        x = F.max_pool2d(x, kernel_size = (1,2), stride = (1,2))
        #print (x.shape)
        x = x.view(-1, 1,x.size(1),x.size(3))
        print (x.shape)
        x = F.relu(self.conv3(x))
        print (x.shape)
        x = F.relu(self.conv4(x))
        print (x.shape)
        #print (x.shape)
        x = F.relu(self.fc1(x))
        #print (x.shape)
        x = self.fc2(x)
        return x


    
    
def train_model(model,criterion, train_input, train_target, test_input, test_target,epochs=250,eta=0.1,mini_batch_size=100):
    # convert inputs to variable 
    train_input_train, train_target_train = train_input, train_target

    train_input, train_target = Variable(train_input), Variable(train_target)
    print ('Training...')
    for e in range(0, epochs):
        sum_loss = 0
        # shuffle data set
        shuffle_indexes = torch.randperm(train_input.size(0))
        train_input = train_input[shuffle_indexes]
        train_target = train_target[shuffle_indexes]
        # We do this with mini-batches
        for b in range(0, train_input.size(0), mini_batch_size):
            output = model.forward(train_input.narrow(0, b, mini_batch_size))
            loss = criterion(output, train_target.narrow(0, b, mini_batch_size))
            sum_loss = sum_loss + loss.data[0]
            model.zero_grad()
            loss.backward()
            for p in model.parameters():
                p.data.sub_(eta * p.grad.data)
        
        print('Train % error:')
        print(compute_nb_errors(model, train_input_train, train_target_train, mini_batch_size=79))
        print('Test % error:')
        print(compute_nb_errors(model, test_input, test_target))
        
    print ('Trained.')
    
def compute_nb_errors(model, input, target, mini_batch_size=100):
    # convert inputs to variable 
    input, target = Variable(input), Variable(target)
    nb_errors = 0
    for b in range(0, input.size(0), mini_batch_size):
        output = model.forward(input.narrow(0, b, mini_batch_size))
        target_output = target.narrow(0, b, mini_batch_size)
        _ ,output_ind= output.max(dim=1, keepdim=True) 
        output_ind = torch.squeeze(output_ind)
        
        error_vec = output_ind!= target_output
        nb_errors+= error_vec.sum().data[0]
          
    return (100*nb_errors/input.size(0))


# In[17]:


model, criterion = LinearPredictor(), nn.CrossEntropyLoss()
eta, mini_batch_size = 1e-5, 79

train_model(model,criterion, train_input, train_target, test_input, test_target,eta=eta,mini_batch_size=mini_batch_size)
print ('Training % error', compute_nb_errors(model, train_input, train_target, mini_batch_size=79) )
print ('Test % error', compute_nb_errors(model, test_input, test_target))


# In[45]:


model, criterion = ShallowConvNetPredictor(), nn.CrossEntropyLoss()
eta, mini_batch_size = 1e-3, 79

train_model(model,criterion, train_input, train_target, test_input, test_target,eta=eta,mini_batch_size=mini_batch_size)
print ('Training % error', compute_nb_errors(model, train_input, train_target, mini_batch_size=79) )
print ('Test % error', compute_nb_errors(model, test_input, test_target))


# In[18]:


model, criterion = ShallowConvNetPredictor_1(), nn.CrossEntropyLoss()
eta, mini_batch_size = 1e-3, 79

train_model(model,criterion, train_input, train_target, test_input, test_target,eta=eta,mini_batch_size=mini_batch_size)
print ('Training % error', compute_nb_errors(model, train_input, train_target, mini_batch_size=79) )
print ('Test % error', compute_nb_errors(model, test_input, test_target))


# In[57]:


model, criterion = ShallowConvNetPredictor_2(), nn.CrossEntropyLoss()
eta, mini_batch_size = 1e-3, 79

train_model(model,criterion, train_input, train_target, test_input, test_target,eta=eta,mini_batch_size=mini_batch_size)
print ('Training % error', compute_nb_errors(model, train_input, train_target, mini_batch_size=79) )
print ('Test % error', compute_nb_errors(model, test_input, test_target))

