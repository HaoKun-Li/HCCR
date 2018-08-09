import torch
import datetime
import time
from models.lossfn import LossFn
from tools.utils import AverageMeter

class AlexNetTrainer(object):

    def __init__(self, lr, train_loader, valid_loader, model, optimizer, scheduler, logger, device):
        self.lr = lr
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.lossfn = LossFn(self.device)
        self.logger = logger
        self.run_count = 0
        self.scalar_info = {}

    def compute_accuracy(self, prob_cls, gt_cls):
        pred_cls = torch.max(prob_cls, 1)[1].squeeze().data.numpy()
        accuracy = float((pred_cls == gt_cls.data.numpy()).astype(int).sum()) / float(gt_cls.size(0))
        return accuracy

    def update_lr(self, epoch):
        """
        update learning rate of optimizers
        :param epoch: current training epoch
        """
        # update learning rate of model optimizer
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.lr

    def train(self, epoch):
        cls_loss_ = AverageMeter()
        accuracy_ = AverageMeter()
        accuracy_valid_ = AverageMeter()


        #训练集作为模型输入
        self.scheduler.step()
        self.model.train()

        for batch_idx, (data, gt_label) in enumerate(self.train_loader):
            data, gt_label = data.to(self.device), gt_label.to(
                self.device)

            cls_pred = self.model(data)
            # compute the loss
            cls_loss = self.lossfn.cls_loss(gt_label, cls_pred)
            accuracy = self.compute_accuracy(cls_pred, gt_label)

            self.optimizer.zero_grad()
            cls_loss.backward()
            self.optimizer.step()

            cls_loss_.update(cls_loss, data.size(0))
            accuracy_.update(accuracy, data.size(0))

        # 验证集作为模型输入
        self.model.eval()

        for batch_idx, (data, gt_label) in enumerate(self.valid_loader):
            data, gt_label = data.to(self.device), gt_label.to(
                self.device)

            cls_pred = self.model(data)
            accuracy_valid = self.compute_accuracy(cls_pred, gt_label)
            accuracy_valid_.update(accuracy_valid, data.size(0))


        #记录数据
        self.scalar_info['cls_loss'] = cls_loss_.avg
        self.scalar_info['accuracy'] = accuracy_.avg
        self.scalar_info['lr'] = self.scheduler.get_lr()[0]

        # if self.logger is not None:
        #     for tag, value in list(self.scalar_info.items()):
        #         self.logger.scalar_summary(tag, value, self.run_count)
        #     self.scalar_info = {}
        # self.run_count += 1

        print("Epoch: {}|===>Train Loss: {:.8f}   Train Accuracy: {:.6f}   valid Accuracy: {:.6f}"
              .format(epoch, cls_loss_.avg, accuracy_.avg, accuracy_valid_.avg))

        return cls_loss_.avg, accuracy_.avg, accuracy_valid_.avg

