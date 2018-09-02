# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from training.AlexNet.config import Config
import torch.nn.init as init

class AlexNet_ST(nn.Module):
    '''AlexNet with Spatial Transformer'''

    def __init__(self):
        super(AlexNet_ST, self).__init__()

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
            nn.MaxPool2d(kernel_size=2, stride=2),
        )  # output shape(256, 3, 3)

        self.conv_small = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )  # output shape(256, 3, 3)



        self.fc = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(256*3*3, 4096, bias=True),
            nn.BatchNorm1d(4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096, bias=True),
            nn.BatchNorm1d(4096),
            nn.ReLU(),
            nn.Linear(4096, self.config.random_size, bias=True),
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.xavier_normal_(m.weight.data)

        # Spatial transformer localization-network
        self.localization = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=11, stride=4,),    # input shape(1, 114, 114)
            nn.BatchNorm2d(16),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(16, 32, kernel_size=5),    # input shape(16, 13, 13)
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(32, 64, kernel_size=5),  # input shape(32, 4, 4)
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
        )

        #  Regressor for the 3 * 2 affine matrix
        self.fc_loc = nn.Sequential(
            nn.Linear(10 * 3 * 3, 32),
            nn.ReLU(True),
            nn.Linear(32, 3 * 2)
        )

        # Initialize the weights/bias with identity transformation
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))


    # Spatila transformer network forward function
    def stn(self, x):
        xs = self.localization(x)
        xs = xs.view(-1, 10 * 3 * 3)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)

        grid = F.affine_grid(theta, x.size())
        x = F.grid_sample(x, grid)

        return x


    def forward(self, x):
        # transform the input
        x = self.stn(x)

        # preform the usual forward pass
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        # x = self.conv_small(x)
        x = x.view(x.size(0), -1)
        output = self.fc(x)                         # flatten the output of conv2 to (batch_size, 256*3*3)
        return output

