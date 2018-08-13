import torch.nn as nn
import torch.nn.functional as F
from training.AlexNet.config import Config
import torch.nn.init as init

class AlexNet(nn.Module):
    '''AlexNet'''

    def __init__(self):
        super(AlexNet, self).__init__()

        self.config = Config()
        self.conv1 = nn.Sequential(                     # input shape(1, 114, 114)
            nn.Conv2d(9, 48, kernel_size=11, stride=4, bias=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.BatchNorm2d(48),
            nn.ReLU(),
        )                                               # output shape(96, 13, 13)
        self.conv2 = nn.Sequential(
            nn.Conv2d(48, 128, kernel_size=5, stride=1, padding=2, bias=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )                                               # output shape(256, 6, 6)

        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 192, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(192),
            nn.ReLU(),
        )                                               # output shape(384, 6, 6)

        self.conv4 = nn.Sequential(
            nn.Conv2d(192, 192, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(192),
            nn.ReLU(),
        )  # output shape(384, 6, 6)

        self.conv5 = nn.Sequential(
            nn.Conv2d(192, 128, kernel_size=3, stride=1, padding=1, bias=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )  # output shape(256, 3, 3)

        self.conv_small = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )  # output shape(256, 3, 3)



        self.fc = nn.Sequential(
            nn.Linear(128*3*3, 2048, bias=True),
            nn.BatchNorm1d(2048),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(2048, 2048, bias=True),
            nn.BatchNorm1d(2048),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(2048, self.config.random_size, bias=True),
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
        # x = self.conv_small(x)
        x = x.view(x.size(0), -1)
        output = self.fc(x)                         # flatten the output of conv2 to (batch_size, 256*3*3)
        return output

