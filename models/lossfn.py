# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class LossFn:
    def __init__(self, device, margin=0.02, lam=2):
        # loss function
        self.margin = margin
        self.lam = lam
        self.device = device
        self.loss_cls = nn.CrossEntropyLoss().to(device)
        self.loss_box = nn.MSELoss().to(device)
        self.loss_landmark = nn.MSELoss().to(device)


    def cls_loss(self, gt_label, pred_label):
        return self.loss_cls(pred_label, gt_label)


    def box_loss(self,gt_label,gt_offset,pred_offset):
        #get the mask element which != 0
        mask = torch.ne(gt_label,0)
        #convert mask to dim index
        chose_index = torch.nonzero(mask)
        chose_index = torch.squeeze(chose_index)
        #only valid element can effect the loss
        valid_gt_offset = gt_offset[chose_index,:]
        valid_pred_offset = pred_offset[chose_index,:]
        valid_pred_offset = torch.squeeze(valid_pred_offset)
        return self.loss_box(valid_pred_offset,valid_gt_offset)


    def landmark_loss(self,gt_label,gt_landmark,pred_landmark):
        mask = torch.eq(gt_label, -2)

        chose_index = torch.nonzero(mask.data)
        chose_index = torch.squeeze(chose_index)

        valid_gt_landmark = gt_landmark[chose_index, :]
        valid_pred_landmark = pred_landmark[chose_index, :]
        return self.loss_landmark(valid_pred_landmark, valid_gt_landmark)

    def Div_loss(self, input):
        b, s = input[0].size()
        diff0 = torch.max(torch.stack((input[1], input[2], input[3]), dim=0), dim=0)[0] - self.margin
        diff1 = torch.max(torch.stack((input[0], input[2], input[3]), dim=0), dim=0)[0] - self.margin
        diff2 = torch.max(torch.stack((input[0], input[1], input[3]), dim=0), dim=0)[0] - self.margin
        diff3 = torch.max(torch.stack((input[0], input[1], input[2]), dim=0), dim=0)[0] - self.margin

        div_loss = (torch.sum(input[0] * diff0) + torch.sum(input[1] * diff1) + torch.sum(input[2] * diff2) +
                    torch.sum(input[3] * diff3)) / b

        return div_loss


    def Dis_loss(self, input):

        def get_maps(a, tx, ty):
            b = tx.size(0)
            maps = torch.zeros([a*a, b], dtype=torch.float32).to(self.device)

            for i in range(a):
                for j in range(a):
                    maps[i * 6 + j] = (i - ty) * (i - ty) + (j - tx)*(j - tx)

            maps = maps.transpose(0, 1)

            return maps


        b, s = input[0].size()
        max_xy = torch.max(input[0], dim=1)[1]
        tx = max_xy % 6
        ty = max_xy / 6
        maps = get_maps(6, tx, ty)
        diff0 = input[0] * maps

        max_xy = torch.max(input[1], dim=1)[1]
        tx = max_xy % 6
        ty = max_xy / 6
        maps = get_maps(6, tx, ty)
        diff1 = input[1] * maps

        max_xy = torch.max(input[2], dim=1)[1]
        tx = max_xy % 6
        ty = max_xy / 6
        maps = get_maps(6, tx, ty)
        diff2 = input[2] * maps

        max_xy = torch.max(input[3], dim=1)[1]
        tx = max_xy % 6
        ty = max_xy / 6
        maps = get_maps(6, tx, ty)
        diff3 = input[3] * maps

        dis_loss = (torch.sum(diff0) + torch.sum(diff1) + torch.sum(diff2) +
                    torch.sum(diff3)) / b

        return dis_loss

    def group_loss(self, input):
        dis_loss = self.Dis_loss(input)
        div_loss = self.lam * self.Div_loss(input)
        total_loss = dis_loss + div_loss
        return total_loss, dis_loss, div_loss
