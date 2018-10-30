# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from training.AlexNet_Attention.config import Config
import torch.nn.init as init
import math


class AlexNet_1(nn.Module):
    '''AlexNet'''

    def __init__(self):
        super(AlexNet_1, self).__init__()

        self.config = Config()
        self.conv1 = nn.Sequential(  # input shape(1, 114, 114)
            nn.Conv2d(1, 96, kernel_size=11, stride=4, bias=True),
            nn.BatchNorm2d(96),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )  # output shape(96, 13, 13)
        self.conv2 = nn.Sequential(
            nn.Conv2d(96, 256, kernel_size=5, stride=1, padding=2, bias=True),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )  # output shape(256, 6, 6)

        self.conv3 = nn.Sequential(
            nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(384),
            nn.ReLU(),
        )  # output shape(384, 6, 6)

        self.conv4 = nn.Sequential(
            nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(384),
            nn.ReLU(),
        )  # output shape(384, 6, 6)

        self.conv5 = nn.Sequential(
            nn.Conv2d(384, 512, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(512),
            nn.ReLU(),
        )  # output shape(512, 3, 3)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        return x


class Classify_part(nn.Module):
    '''Classify_part'''

    def __init__(self):
        super(Classify_part, self).__init__()

        self.config = Config()
        self.conv1 = nn.Sequential(  # input shape(1, 48, 48)
            nn.Conv2d(1, 96, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(96),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )  # output shape(96, 24, 24)
        self.conv2 = nn.Sequential(
            nn.Conv2d(96, 128, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )  # output shape(128, 12, 12)

        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 160, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(160),
            nn.ReLU(),
        )  # output shape(160, 12, 12)

        self.conv4 = nn.Sequential(
            nn.Conv2d(160, 256, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )  # output shape(256, 6, 6)

        self.conv5 = nn.Sequential(
            nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(384),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )  # output shape(384, 3, 3)
        self.conv_f1 = nn.Sequential(
            nn.Conv2d(384, 1024, kernel_size=1, stride=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
        )
        self.gap = nn.AdaptiveAvgPool2d(1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv_f1(x)
        x = self.gap(x)
        # print(self.conv_f1[1].weight.data)
        return x


class Channel_group_sub(nn.Module):
    '''Channel_group'''

    def __init__(self):
        super(Channel_group_sub, self).__init__()

        self.group = nn.Sequential(
            nn.Linear(512, 512, bias=False),
            nn.Tanh(),
            nn.Linear(512, 512, bias=False),
            nn.Sigmoid(),
        )

        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.xavier_normal_(m.weight.data)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x, y):
        b, c, _, _ = y.size()
        x = self.group(x).view(b, c, 1, 1)  # d_i(x)
        # print('after group: ', torch.max(x), torch.min(x))
        mask = (x * y).view(b, c, -1).transpose(1, 2)
        # print('before pool: ', mask.size(), torch.max(mask), torch.min(mask))
        mask = self.avgpool(mask)
        # print(mask.size())     # (64, 36, 1)
        # print('after pool: ', mask.size(), torch.max(mask), torch.min(mask), torch.mean(mask))

        mask = self.sigmoid(mask).view(b, -1)
        # mask = self.tanh(mask).view(b, -1)
        # print('after Tanh: ', torch.max(mask[10]), torch.min(mask[10]), torch.mean(mask[10]))

        # mask = torch.log(mask)
        # print('after Log: ', torch.max(mask[10]), torch.min(mask[10]), torch.mean(mask[10]))
        # print(mask[10])

        mask = self.softmax(mask)
        # print('after softmax: ', torch.max(mask[10]), torch.min(mask[10]), torch.mean(mask[10]))
        # print(mask[10])

        return mask


class Attention_CNN(nn.Module):
    '''MA_CNN'''

    def __init__(self):
        super(Attention_CNN, self).__init__()

        self.config = Config()
        self.alexnet_1 = AlexNet_1()

        self.upsample = nn.Sequential(
            nn.Conv2d(512, 1, kernel_size=1, stride=1, bias=False),
            nn.Upsample(scale_factor=16, mode='bilinear', align_corners=True),
        )

        self.sumConv = nn.Conv2d(512, 1, kernel_size=1, stride=1, bias=False)

        self.cla_part1 = Classify_part()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.xavier_normal_(m.weight.data)

        for param in self.upsample.parameters():
            param.requires_grad = False
        self.upsample[0].weight.data.fill_(1.0)

        for param in self.sumConv.parameters():
            param.requires_grad = False
        self.sumConv.weight.data.fill_(1.0)

    def forward(self, x):
        x = self.alexnet_1(x)  # x : torch.Size([64, 512, 6, 6])

        mask = self.sumConv(x)
        mask = mask.view(mask.size(0), -1)

        att = self.upsample(x)

        return x, mask


class Channel_group(nn.Module):
    '''Channel_group'''

    def __init__(self):
        super(Channel_group, self).__init__()

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.channelgroup_1 = Channel_group_sub()
        self.channelgroup_2 = Channel_group_sub()

    def forward(self, x):  # x : torch.Size([64, 512, 6, 6])
        x_gap = self.gap(x).view(x.size(0), -1)

        x_group_1 = self.channelgroup_1(x_gap, x)
        x_group_2 = self.channelgroup_2(x_gap, x)
        mask = [x_group_1, x_group_2]   # mask[0] : torch.Size([64, 36])

        return mask


class Classify(nn.Module):
    '''Classify'''

    def __init__(self):
        super(Classify, self).__init__()

        self.config = Config()
        self.conv_f1 = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=1, stride=1),
            # nn.BatchNorm2d(1024),
            # nn.ReLU()
        )
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.Classify_1 = Classify_part()
        self.fc = nn.Sequential(
            nn.Linear(2048, self.config.random_size)
            )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, part, x_ori):
        x_ori = self.conv_f1(x_ori)
        x_ori = self.gap(x_ori).view(x_ori.size(0), -1)
        x_part1 = self.Classify_1(part).view(x_ori.size(0), -1)
        x = torch.cat((x_ori, x_part1), 1)

        # print('part1: ', torch.max(x_part1[10]), torch.min(x_part1[10]), torch.mean(x_part1[10]))
        # print('part2: ', torch.max(x_part2[10]), torch.min(x_part2[10]), torch.mean(x_part2[10]))
        # print('ori: ', torch.max(x_ori[10]), torch.min(x_ori[10]), torch.mean(x_ori[10]))
        # x = torch.cat((x_ori, x_ori, x_ori, x_ori, x_ori), 1)
        # x = torch.cat((x_part1, x_part2, x_part3, x_part4), 1)
        output = self.fc(x)

        return output

