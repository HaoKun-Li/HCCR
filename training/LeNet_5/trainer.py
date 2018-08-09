import torch
import datetime
import time
from models.lossfn import LossFn
from tools.utils import AverageMeter

class LeNet_5Trainer(object):

    def __init__(self, lr, train_loader, valid_x, valid_y, model, optimizer, scheduler, logger, device):
        self.lr = lr
        self.train_loader = train_loader
        self.valid_x = valid_x
        self.valid_y = valid_y
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

    def train(self, epoch):
        cls_loss_ = AverageMeter()
        accuracy_ = AverageMeter()
        accuracy_valid_ = AverageMeter()

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

            if batch_idx % 50 == 0:
                valid_pred = self.model(self.valid_x)
                accuracy_valid = self.compute_accuracy(valid_pred, self.valid_y)
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tTrain Loss: {:.6f}\tTrain Accuracy: {:.6f}\tValid Accuracy: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(self.train_loader.dataset),
                    100. * batch_idx / len(self.train_loader), cls_loss.item(), accuracy, accuracy_valid))

                accuracy_valid_.update(accuracy_valid, data.size(0))


        self.scalar_info['cls_loss'] = cls_loss_.avg
        self.scalar_info['accuracy'] = accuracy_.avg
        self.scalar_info['lr'] = self.lr

        # if self.logger is not None:
        #     for tag, value in list(self.scalar_info.items()):
        #         self.logger.scalar_summary(tag, value, self.run_count)
        #     self.scalar_info = {}
        # self.run_count += 1

        print("|===>Loss: {:.4f}   Train Accuracy: {:.6f}   valid Accuracy: {:.6f}".format(cls_loss_.avg, accuracy_.avg, accuracy_valid_.avg))

        valid_output = self.model(self.valid_x[:10])
        pred_y = torch.max(valid_output, 1)[1].squeeze().data.numpy()
        print(pred_y, "prediction number")
        print(self.valid_y[:10].numpy(), "real number")

        return cls_loss_.avg, accuracy_.avg, accuracy_valid_.avg
