#!/usr/bin/env python
# coding: utf-8

# In[1]:


'''
Reference: https://blog.csdn.net/Gilgame/article/details/85056344
'''


# In[4]:


import torch
import torch.nn as nn

# cuda
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# In[33]:


class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()
        self.conv1 = torch.nn.Sequential(  #input_size = 32*32*3
            torch.nn.Conv2d(in_channels = 3, 
                            out_channels = 6,
                            kernel_size = 3,
                            stride = 1,
                            padding = 1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size = 2,
                               stride = 2)  #output_size = 16*16*6
        )
        
        self.conv2 = torch.nn.Sequential(  #input_size = 16*16*6
            torch.nn.Conv2d(6, 16, 3, 1, 1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2, 2)  #output_size = 8*8*16 
        )
        self.conv3 = torch.nn.Sequential(  #input_size = 8*8*16
            torch.nn.Conv2d(16, 32, 3, 1, 1),
            torch.nn.ReLU()  #output_size = 8*8*32
        )
        self.conv4 = torch.nn.Sequential(  #input_size = 8*8*32
            torch.nn.Conv2d(32, 32, 3, 1, 1), 
            torch.nn.ReLU()  #output_size = 8*8*32
        )
        self.conv5 = torch.nn.Sequential(  #input_size = 8*8*32
            torch.nn.Conv2d(32, 64, 3, 1, 1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2, 2)   #output_size = 4*4*64
        )
        
        # FCL
        self.dense = torch.nn.Sequential(
            torch.nn.Linear(1024, 512),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(512, 512),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(512, 10) # final number of classes
        )
        
    def forward(self, x): # 正向传播过程
        x = self.conv1(x)
        print(x.shape)
        x = self.conv2(x)
        print(x.shape)
        x = self.conv3(x)
        print(x.shape)
        x = self.conv4(x)
        print(x.shape)
        x = self.conv5(x)
        print(x.shape)
            
        x = x.view(x.size(0), -1)
        out = self.dense(x)
            
        return out
            

