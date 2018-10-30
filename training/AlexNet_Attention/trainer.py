# -*- coding: utf-8 -*-
import torch
import datetime
import time
from models.lossfn import LossFn
from tools.utils import AverageMeter
import cv2
import numpy as np
from training.AlexNet_Attention.config import Config
import torchvision


# import matplotlib.pyplot as plt


# this function is to get four part locations which is the output of our designed part network
def part_box(mask):
    part_max = torch.argmax(mask, dim=1)
    part = torch.FloatTensor(2, 64).to(part_max.device)

    part[0] = part_max % 6
    part[1] = part_max / 6

    return part


def part_box_two_parts(mask):
    part1 = torch.argmax(mask[0], dim=1)
    part2 = torch.argmax(mask[1], dim=1)
    part = torch.FloatTensor(2, 2, 64).to(part1.device)

    part[0][0] = part1 % 6
    part[0][1] = part1 / 6
    part[1][0] = part2 % 6
    part[1][1] = part2 / 6

    return part


def get_part(img, parts):
    b, c, h, w = img.size()
    img_parts = []
    parts = parts.cpu().numpy()

    parts[0] = parts[0] * h / 6 + 9
    parts[1] = parts[1] * h / 6 + 9
    l = 24

    boxs = np.array([np.maximum(0, (parts[0] - l)), np.maximum(0, (parts[1] - l)),
                        np.minimum(w, (parts[0] + l)), np.minimum(h, (parts[1] + l))])

    boxs = np.array(boxs).transpose(1, 0).astype(np.int32)
    img = img.numpy().reshape(b, h, w)

    for i in range(b):
        box = boxs[i]
        img_part = cv2.resize(img[i][box[1]: box[3], box[0]: box[2]], (48, 48))

        img_parts.append(img_part)

    img_parts = np.array(img_parts)  # (64, 48, 48)
    return img_parts, parts


def get_part_two_parts(img, parts):
    b, c, h, w = img.size()
    img_parts = []
    boxs = [[], []]
    parts = parts.cpu().numpy()

    for i in range(2):
        parts[i][0] = parts[i][0] * h / 6 + 9
        parts[i][1] = parts[i][1] * h / 6 + 9
        l = 24

        boxs[i] = np.array([np.maximum(0, (parts[i][0] - l)), np.maximum(0, (parts[i][1] - l)),
                            np.minimum(w, (parts[i][0] + l)), np.minimum(h, (parts[i][1] + l))])

    boxs = np.array(boxs).transpose(2, 0, 1).astype(np.int32)
    img = img.numpy().reshape(b, h, w)

    for i in range(b):
        box = boxs[i]
        img_part = [[] for i in range(2)]
        for j in range(2):
            img_part[j] = cv2.resize(img[i][box[j][1]: box[j][3], box[j][0]: box[j][2]], (48, 48))

        img_parts.append(img_part)

    img_parts = np.array(img_parts).transpose(1, 0, 2, 3)  # (2, 64, 48, 48)
    return img_parts, parts


class AlexNetTrainer(object):

    def __init__(self, lr, train_loader, valid_loader, model_1, model_2, optimizer_1, optimizer_2,
                 scheduler_1, scheduler_2, logger, device):
        self.lr = lr
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.model_1 = model_1
        self.optimizer_1 = optimizer_1
        self.scheduler_1 = scheduler_1
        self.model_2 = model_2
        self.optimizer_2 = optimizer_2
        self.scheduler_2 = scheduler_2
        self.device = device
        self.lossfn = LossFn(self.device, lam=2)
        self.logger = logger
        self.run_count = 0
        self.scalar_info = {}
        self.config = Config()
        # hook
        # self.handle = self.model_2.channelgroup_2.group[0].register_backward_hook(self.for_hook)

    def compute_accuracy(self, prob_cls, gt_cls):
        pred_cls = torch.max(prob_cls, 1)[1].squeeze()
        accuracy = float((pred_cls == gt_cls).sum()) / float(gt_cls.size(0))
        return accuracy

    def update_lr(self, epoch):
        """
        update learning rate of optimizers
        :param epoch: current training epoch
        """
        # update learning rate of model optimizer
        for param_group in self.optimizer_1.param_groups:
            param_group['lr'] = self.lr
        for param_group in self.optimizer_2.param_groups:
            param_group['lr'] = self.lr

    def show_image_grid(self, img_origin, img_parts, parts, epoch):
        print(img_parts.size())
        torchvision.utils.save_image(img_origin, self.config.save_path + '/img_origin_grid.jpg', nrow=8, padding=2,
                                     normalize=True,
                                     range=(0, 1))

        torchvision.utils.save_image(img_parts, self.config.save_path + '/img_parts' + '_grid.jpg',
                                    nrow=8, padding=2, normalize=True,
                                    range=(0, 1))

        print(type(parts))
        print(parts.shape)

        grid = torchvision.utils.make_grid(img_origin, nrow=8, padding=2, normalize=True,
                                           range=(0, 1))

        image_size = 114
        parts = parts.transpose((1, 0))

        # save grid and parts as numpy
        grid = grid.cpu().numpy().transpose((1, 2, 0))
        np.save(self.config.save_path + '/grid.npy', grid)
        np.save(self.config.save_path + '/parts.npy', parts)

        img = cv2.cvtColor(grid * 255, cv2.COLOR_RGB2BGR)
        l = 24
        colors = [(229, 187, 129), (161, 23, 21), (34, 8, 7), (118, 77, 57)]
        for i in range(parts.shape[0]):
            box = np.array([np.maximum(0, (parts[i, 0] - l)), np.maximum(0, (parts[i, 1] - l)),
                            np.minimum(image_size, (parts[i, 0] + l)),
                            np.minimum(image_size, (parts[i, 1] + l))])
            cv2.rectangle(img, (int(box[0] + i % 8 * (2 + image_size)), int(box[1]) + i // 8 * (2 + image_size)),
                            (int(box[2] + i % 8 * (2 + image_size)), int(box[3]) + i // 8 * (2 + image_size)),
                            colors[0], 2)

        cv2.imwrite(self.config.save_path + '/plt_image_boxs_epoch' + str(epoch) + '.jpg', img)  # 保存图片

        return

    def show_mask(self, mask, epoch):
        img = (1 - (mask - torch.min(mask)) / (torch.max(mask) - torch.min(mask)))
        img = torchvision.utils.make_grid(img.view(64, 1, 6, 6), nrow=8, padding=2, normalize=True,
                                              range=(0, 1)).cpu().numpy().transpose((1, 2, 0))
        img = cv2.cvtColor(img * 255, cv2.COLOR_RGB2BGR)
        cv2.imwrite(self.config.save_path + '/mask_' + 'epoch' + str(epoch) + '.jpg', img)  # 保存图片

        return

    def for_hook(self, module, input_grad, output_grad):
        print('\r\nhook:\r\n')
        print(len(input_grad))
        print(len(output_grad))
        print(input_grad[0].size())
        print(input_grad[1].size())
        print(output_grad[0].size())
        print(input_grad[0][5][5:10])
        print(output_grad[0][5][5:10])

    def train(self, epoch):
        cls_loss_ = AverageMeter()
        accuracy_ = AverageMeter()
        accuracy_valid_ = AverageMeter()

        # 训练集作为模型输入
        self.scheduler_1.step()
        self.scheduler_2.step()
        self.model_1.train()
        self.model_2.train()

        for batch_idx, (data, gt_label) in enumerate(self.train_loader):

            data, gt_label = data.to(self.device), gt_label.to(
                self.device)
            x, mask = self.model_1(data)

            # test
            # print(self.model_1.alexnet_1.conv1[0].weight.data)
            # print(self.model_2.channelgroup_2.group[0].weight.data[5][5:10])
            # print(self.model_3.Classify_1.conv1[0].weight.data)

            # test

            with torch.no_grad():
                parts = part_box(mask)
                img_parts, parts = get_part(data.cpu(), parts)  # (1, 64, 48, 48)
                img_parts = torch.from_numpy(img_parts).view(img_parts.shape[0], 1, 48, 48).to(
                    self.device)  # view(64, 1, 48, 48)

                if (epoch == 1 or epoch == 5 or epoch == 10 or epoch == 15) and batch_idx == 1:
                    self.show_image_grid(data, img_parts, parts, epoch)
                    self.show_mask(mask, epoch)
                    print('save image and parts in result: ' + self.config.save_path)
                    print('epoch: ' + str(epoch))
                    print('batch_idx: ' + str(batch_idx))

            cls_pred = self.model_2(img_parts, x)

            # compute the loss
            cls_loss = self.lossfn.cls_loss(gt_label, cls_pred)
            accuracy = self.compute_accuracy(cls_pred, gt_label)

            if epoch >= 0:
                self.optimizer_1.zero_grad()
                self.optimizer_2.zero_grad()
                cls_loss.backward()
                self.optimizer_1.step()
                self.optimizer_2.step()

            cls_loss_.update(cls_loss.item(), data.size(0))
            accuracy_.update(accuracy, data.size(0))

            if batch_idx % 2000 == 1:
                print('batch_idx: ', batch_idx)
                print('Cls loss: ', cls_loss.item())

        # 验证集作为模型输入
        with torch.no_grad():
            self.model_1.eval()
            self.model_2.eval()

            for batch_idx, (data, gt_label) in enumerate(self.valid_loader):
                data, gt_label = data.to(self.device), gt_label.to(
                    self.device)

                x, mask = self.model_1(data)

                parts = part_box(mask)
                img_parts, parts = get_part(data.cpu(), parts)  # (4, 64, 48, 48)

                img_parts = torch.from_numpy(img_parts).view(img_parts.shape[0], 1, 48, 48).to(self.device)

                cls_pred = self.model_2(img_parts, x)

                accuracy_valid = self.compute_accuracy(cls_pred, gt_label)
                accuracy_valid_.update(accuracy_valid, data.size(0))

            # 记录数据
            self.scalar_info['cls_loss'] = cls_loss_.avg
            self.scalar_info['accuracy'] = accuracy_.avg
            self.scalar_info['lr'] = self.scheduler_1.get_lr()[0]

            # if self.logger is not None:
            #     for tag, value in list(self.scalar_info.items()):
            #         self.logger.scalar_summary(tag, value, self.run_count)
            #     self.scalar_info = {}
            # self.run_count += 1

        print("\r\nEpoch: {}|===>Train Loss: {:.8f}   Train Accuracy: {:.6f}   valid Accuracy: {:.6f}\r\n"
              .format(epoch, cls_loss_.avg, accuracy_.avg, accuracy_valid_.avg))

        return cls_loss_.avg, accuracy_.avg, accuracy_valid_.avg

