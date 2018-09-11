# -*- coding: utf-8 -*-
import torch
import datetime
import time
from models.lossfn import LossFn
from tools.utils import AverageMeter
import cv2
import numpy as np
from training.AlexNet_MA.config import Config
import torchvision


# this function is to get four part locations which is the output of our designed part network
def part_box(mask):
    part1 = torch.argmax(mask[0], dim=1)
    part2 = torch.argmax(mask[1], dim=1)
    part3 = torch.argmax(mask[2], dim=1)
    part4 = torch.argmax(mask[3], dim=1)
    part = torch.FloatTensor(4, 2, 64).to(part1.device)

    part[0][0] = part1 % 6
    part[0][1] = part1 / 6
    part[1][0] = part2 % 6
    part[1][1] = part2 / 6
    part[2][0] = part3 % 6
    part[2][1] = part3 / 6
    part[3][0] = part4 % 6
    part[3][1] = part4 / 6

    return part


def get_part(img, parts):
    b, c, h, w = img.size()
    img_parts = []
    boxs = [[], [], [], []]
    parts = parts.cpu().numpy()

    for i in range(4):
        parts[i][0] = parts[i][0] * h / 6 + 9
        parts[i][1] = parts[i][1] * h / 6 + 9
        l = 16

        boxs[i] = np.array([np.maximum(0, (parts[i][0] - l)), np.maximum(0, (parts[i][1] - l)),
                            np.minimum(w, (parts[i][0] + l)), np.minimum(h, (parts[i][1] + l))])

    boxs = np.array(boxs).transpose(2, 0, 1).astype(np.int32)
    img = img.numpy().reshape(b, h, w)

    for i in range(b):
        box = boxs[i]
        img_part = [[] for i in range(4)]
        for j in range(4):
            img_part[j] = cv2.resize(img[i][box[j][1]: box[j][3], box[j][0]: box[j][2]], (48, 48))

        img_parts.append(img_part)

    img_parts = np.array(img_parts).transpose(1, 0, 2, 3)  # (4, 64, 48, 48)
    return img_parts


class AlexNetTrainer(object):

    def __init__(self, lr, train_loader, valid_loader, model_1, model_2, model_3, optimizer_1, optimizer_2, optimizer_3,
                 scheduler_1, scheduler_2, scheduler_3, logger, device):
        self.lr = lr
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.model_1 = model_1
        self.optimizer_1 = optimizer_1
        self.scheduler_1 = scheduler_1
        self.model_2 = model_2
        self.optimizer_2 = optimizer_2
        self.scheduler_2 = scheduler_2
        self.model_3 = model_3
        self.optimizer_3 = optimizer_3
        self.scheduler_3 = scheduler_3
        self.device = device
        self.lossfn = LossFn(self.device)
        self.logger = logger
        self.run_count = 0
        self.scalar_info = {}
        self.config = Config()

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


    def show_image_grid(self, img_origin, img_parts):
        print(img_parts.size())
        torchvision.utils.save_image(img_origin, self.config.save_path+'/img_origin_grid.jpg', nrow=8, padding=2, normalize=True,
                                     range=(0, 1))
        torchvision.utils.save_image(img_parts[0], self.config.save_path+'/img_parts0_grid.jpg', nrow=8, padding=2, normalize=True,
                                     range=(0, 1))
        torchvision.utils.save_image(img_parts[1], self.config.save_path+'/img_parts1_grid.jpg', nrow=8, padding=2, normalize=True,
                                     range=(0, 1))
        torchvision.utils.save_image(img_parts[2], self.config.save_path+'/img_parts2_grid.jpg', nrow=8, padding=2, normalize=True,
                                     range=(0, 1))
        torchvision.utils.save_image(img_parts[3], self.config.save_path+'/img_parts3_grid.jpg', nrow=8, padding=2, normalize=True,
                                     range=(0, 1))

        return


    def train(self, epoch):
        cls_loss_ = AverageMeter()
        accuracy_ = AverageMeter()
        accuracy_valid_ = AverageMeter()

        # 训练集作为模型输入
        self.scheduler_1.step()
        self.scheduler_2.step()
        self.scheduler_3.step()
        self.model_1.train()
        self.model_2.train()
        self.model_3.train()

        for batch_idx, (data, gt_label) in enumerate(self.train_loader):

            data, gt_label = data.to(self.device), gt_label.to(
                self.device)
            x, att = self.model_1(data)
            mask = self.model_2(x)

            # test
            # print(self.model_1.alexnet_1.conv1[0].weight.data)
            # print(self.model_2.channelgroup_2.group[0].weight.data[5][5:10])
            # print(self.model_3.Classify_1.conv1[0].weight.data)

            g_loss, dis_loss, div_loss = self.lossfn.group_loss(mask)

            self.optimizer_2.zero_grad()
            g_loss.backward(retain_graph=True)
            self.optimizer_2.step()
            # test

            with torch.no_grad():
                parts = part_box(mask)
                img_parts = get_part(data.cpu(), parts)  # (4, 64, 48, 48)
                img_parts = torch.from_numpy(img_parts).view(4, 64, 1, 48, 48).to(self.device)

                if epoch == 15:
                    self.show_image_grid(data, img_parts)
                    print('save image and parts in result: '+ self.config.save_path)


            cls_pred = self.model_3(img_parts, x)

            # compute the loss
            cls_loss = self.lossfn.cls_loss(gt_label, cls_pred)
            accuracy = self.compute_accuracy(cls_pred, gt_label)

            self.optimizer_1.zero_grad()
            self.optimizer_3.zero_grad()
            cls_loss.backward()
            self.optimizer_1.step()
            self.optimizer_3.step()

            cls_loss_.update(cls_loss.item(), data.size(0))
            accuracy_.update(accuracy, data.size(0))

            if batch_idx % 2000 == 10:
                print(batch_idx)
                print(g_loss.item(), dis_loss.item(), div_loss.item())
                print(cls_loss.item())
                print(self.model_2.channelgroup_2.group[0].weight.data[5][5:10])
            # print(batch_idx)
            # print(g_loss.item(), dis_loss.item(), div_loss.item())
            # print(cls_loss.item())
            # print(self.model_2.channelgroup_2.group[0].weight.data[5][5:10])

        # 验证集作为模型输入
        with torch.no_grad():
            self.model_1.eval()
            self.model_2.eval()
            self.model_3.eval()

            for batch_idx, (data, gt_label) in enumerate(self.valid_loader):
                data, gt_label = data.to(self.device), gt_label.to(
                    self.device)

                x, att = self.model_1(data)
                mask = self.model_2(x)

                parts = part_box(mask)
                img_parts = get_part(data.cpu(), parts)  # (4, 64, 48, 48)

                img_parts = torch.from_numpy(img_parts).view(4, 64, 1, 48, 48).to(self.device)

                cls_pred = self.model_3(img_parts, x)

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

