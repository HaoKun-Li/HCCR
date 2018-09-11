# -*- coding: utf-8 -*-
import torch.nn as nn
import torch.nn.functional as F
from training.AlexNet.config import Config
import torch.nn.init as init

class AlexNet_gap(nn.Module):
    '''AlexNet with gap'''

    def __init__(self):
        super(AlexNet_gap, self).__init__()

        self.config = Config()
        self.conv1 = nn.Sequential(                     # input shape(1, 114, 114)
            nn.Conv2d(1, 96, kernel_size=11, stride=4, bias=True),
            nn.BatchNorm2d(96),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )                                               # output shape(96, 13, 13)
        self.conv2 = nn.Sequential(
            nn.Conv2d(96, 256, kernel_size=5, stride=1, padding=2, bias=True),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )                                               # output shape(256, 6, 6)

        self.conv3 = nn.Sequential(
            nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(384),
            nn.ReLU(),
        )                                               # output shape(384, 6, 6)

        self.conv4 = nn.Sequential(
            nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(384),
            nn.ReLU(),
        )  # output shape(384, 6, 6)

        self.conv5 = nn.Sequential(
            nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )  # output shape(256, 3, 3)

        self.conv_f1 = nn.Conv2d(256, 4096, kernel_size=1, stride=1)
        self.gap = nn.AdaptiveAvgPool2d(1)

        self.fc = nn.Sequential(
            nn.BatchNorm1d(4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096, self.config.random_size, bias=True),
        )


        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.xavier_normal_(m.weight.data)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)

        x = self.conv_f1(x)
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        output = self.fc(x)
        return output                        # flatten the output of conv2 to (batch_size, 256*3*3)

