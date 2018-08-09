import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

class LeNet_5(nn.Module):
    '''LeNet_5'''

    def __init__(self):
        super(LeNet_5, self).__init__()

        self.conv1 = nn.Sequential(                     # input shape(1, 28, 28)
            nn.Conv2d(1, 6, kernel_size=5, stride=1, bias=True),
            nn.MaxPool2d(kernel_size=2),
        )                                               # output shape(6, 12, 12)
        self.conv2 = nn.Sequential(
            nn.Conv2d(6, 16, kernel_size=5, stride=1, bias=True),
            nn.MaxPool2d(kernel_size=2),
        )                                               # output shape(16, 4, 4)

        self.fc = nn.Sequential(
            nn.Linear(16*4*4, 120, bias=True),
            nn.Linear(120, 84, bias=True),
            nn.Linear(84, 10, bias=True),
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.xavier_normal_(m.weight.data)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        output = self.fc(x)                         # flatten the output of conv2 to (batch_size, 16*4*4)
        return output

