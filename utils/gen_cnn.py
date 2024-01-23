#!/usr/bin/env python
# coding: utf-8

import torch
from torch import nn
import torch.nn.functional as F

class CNNMnist(nn.Module):
    def __init__(self, num_channels, num_classes):
        super(CNNMnist, self).__init__()
        self.conv1 = nn.Conv2d(num_channels, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, num_classes)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, x.shape[1]*x.shape[2]*x.shape[3])
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return x

class CNNMnistSglLyr(nn.Module):
    def __init__(self, num_channels, num_classes):
        super(CNNMnistSglLyr, self).__init__()
        self.conv1 = nn.Conv2d(num_channels, 10, kernel_size=5)
        self.fc2 = nn.Linear(1440, num_classes)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = x.view(-1, x.shape[1]*x.shape[2]*x.shape[3])
        x = self.fc2(x)
        return x

class CNNCifar(nn.Module):
    def __init__(self, num_channels, num_classes):
        super(CNNCifar, self).__init__()
        self.conv1 = nn.Conv2d(num_channels, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x))) 
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, x.shape[1]*x.shape[2]*x.shape[3])
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class CNNCifar1(nn.Module):
    def __init__(self, num_channels, num_classes):
        super(CNNCifar1, self).__init__()        
        self.conv1 = nn.Conv2d(num_channels, 48, 4)
        self.conv2 = nn.Conv2d(48, 100, 4)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(100*5*5, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(p=.4)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x))) 
        x = self.dropout(x)
        x = self.pool(F.relu(self.conv2(x))) 
        x = self.dropout(x)
        x = x.view(-1, x.shape[1]*x.shape[2]*x.shape[3])
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        x = self.fc3(x)
        return x

class CNNCifar2(nn.Module):
    def __init__(self, num_channels, num_classes):
        super(CNNCifar2, self).__init__()
        self.conv1 = nn.Conv2d(num_channels, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64*4*4, 512)
        self.fc2 = nn.Linear(512, 64)
        self.fc3 = nn.Linear(64, num_classes)
        self.dropout = nn.Dropout(p=.5)
        
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x))) 
        x = self.pool(F.relu(self.conv2(x))) 
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, x.shape[1]*x.shape[2]*x.shape[3])
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        x = self.fc3(x)
        return x


