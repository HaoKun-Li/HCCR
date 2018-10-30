# -*- coding: utf-8 -*-
import torch.nn as nn
import torch.nn.functional as F
from training.FastCNN.config import Config
import torch.nn.init as init

class FastCNN(nn.Module):
    '''
    FastCNN from
    Building Fast and Compact Convolutional Neural Networks for Oﬄine Handwritten Chinese Character Recognition
    '''


    def __init__(self):
        super(FastCNN, self).__init__()

        self.config = Config()
        self.conv1 = nn.Sequential(                     # input shape(1, 96, 96)
            nn.Conv2d(1, 96, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(96),
            nn.PReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )                                               # output shape(96, 48, 48)
        self.conv2 = nn.Sequential(
            nn.Conv2d(96, 128, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(128),
            nn.PReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )                                               # output shape(128, 24, 24)

        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 160, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(160),
            nn.PReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )                                               # output shape(160, 12, 12)

        self.conv4 = nn.Sequential(
            nn.Conv2d(160, 256, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(256),
            nn.PReLU(),
        )  # output shape(256, 12, 12)

        self.conv5 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(256),
            nn.PReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )  # output shape(256, 6, 6)

        self.conv6 = nn.Sequential(
            nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(384),
            nn.PReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )  # output shape(384, 3, 3)


        self.fc = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(384*3*3, 1024, bias=True),
            nn.BatchNorm1d(1024),
            nn.PReLU(),
            nn.Linear(1024, self.config.random_size, bias=True),
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.xavier_normal_(m.weight.data)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = x.view(x.size(0), -1)
        feature = x
        output = self.fc(x)                         # flatten the output of conv2 to (batch_size, 384*3*3)
        return output, feature

