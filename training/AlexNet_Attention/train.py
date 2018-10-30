# -*- coding: utf-8 -*-
import torch
from torchvision import transforms
import numpy as np
import sys
import os

sys.path.append('../../')

from models.AlexNet import AlexNet
from models.AlexNet_ST import AlexNet_ST
from models.AlexNet_SE import AlexNet_SE
# import models.MA_CNN as model
import models.Attention_CNN as model
from models.AlexNet_gap import AlexNet_gap
from models.AlexNet_gap_SE import AlexNet_gap_SE

from training.AlexNet_Attention.trainer import AlexNetTrainer
from training.AlexNet_Attention.config import Config
from tools.logger import Logger
from checkpoint import CheckPoint
import os
import tools.imagedb as imagedb
import time
import multiprocessing

if __name__ == '__main__':
    multiprocessing.freeze_support()
    config = Config()
    if not os.path.exists(config.save_path):
        os.makedirs(config.save_path)

    os.environ['CUDA_VISIBLE_DEVICES'] = config.GPU
    use_cuda = config.use_cuda and torch.cuda.is_available()
    torch.manual_seed(config.manualSeed)
    torch.cuda.manual_seed(config.manualSeed)
    device = torch.device("cuda" if use_cuda else "cpu")
    torch.backends.cudnn.benchmark = True

    imagedb.preprocess_gnt()
    train_annotation_path = os.path.join(config.dataPath, 'train_png_annotation.txt')
    valid_annotation_path = os.path.join(config.dataPath, 'valid_png_annotation.txt')
    train_prefix_path = os.path.join(config.dataPath, 'train_png')
    valid_prefix_path = os.path.join(config.dataPath, 'valid_png')

    # 记录字符集
    imagedb.write_char_set()

    # #Set dataloader
    kwargs = {'num_workers': config.nThreads, 'pin_memory': True} if use_cuda else {}
    transform_train = transforms.Compose([
        transforms.RandomCrop(size=(114, 114), padding=8),
        transforms.ToTensor()
    ])
    transform_valid = transforms.Compose([
        transforms.ToTensor()
    ])
    train_loader = torch.utils.data.DataLoader(
        dataset=imagedb.HCDataset(train_annotation_path, prefix_path=train_prefix_path, transform=transform_train),
        batch_size=config.batchSize,
        shuffle=True,
        drop_last=True,
        **kwargs)

    valid_loader = torch.utils.data.DataLoader(
        dataset=imagedb.HCDataset(valid_annotation_path, prefix_path=valid_prefix_path, transform=transform_valid),
        batch_size=config.batchSize,
        shuffle=True,
        drop_last=True,
        **kwargs)

    print(len(train_loader.dataset))
    print(len(valid_loader.dataset))

    # Set model
    model_1 = model.Attention_CNN()
    model_2 = model.Classify()

    para_1 = sum([np.prod(list(p.size())) for p in model_1.parameters()])
    para_2 = sum([np.prod(list(p.size())) for p in model_2.parameters()])
    print('Model_1 {} : params: {:4f}M'.format(model_1._get_name(), para_1 * 4 / 1000 / 1000))
    print('Model_2 {} : params: {:4f}M'.format(model_2._get_name(), para_2 * 4 / 1000 / 1000))
    model_1 = model_1.to(device)
    model_2 = model_2.to(device)

    # Set checkpoint
    checkpoint = CheckPoint(config.save_path)

    # Set optimizer
    optimizer_1 = torch.optim.Adam(filter(lambda p: p.requires_grad, model_1.parameters()), lr=0.01)
    scheduler_1 = torch.optim.lr_scheduler.MultiStepLR(optimizer_1, milestones=config.step, gamma=0.1)

    optimizer_2 = torch.optim.Adam(filter(lambda p: p.requires_grad, model_2.parameters()), lr=0.01)
    scheduler_2 = torch.optim.lr_scheduler.MultiStepLR(optimizer_2, milestones=config.step, gamma=0.1)

    # Set trainer
    logger = Logger(config.save_path)
    trainer = AlexNetTrainer(config.lr, train_loader, valid_loader,
                             model_1, model_2,
                             optimizer_1, optimizer_2,
                             scheduler_1, scheduler_2,
                             logger, device)

    print(model_1)
    print(model_2)
    epoch_dict = 1

    # load_model
    # model_dict_1, optimizer_dict_1, epoch_dict = checkpoint.load_checkpoint(
    #     os.path.join(checkpoint.save_path, 'checkpoint_030_model_1.pth'))
    # model_1.load_state_dict(model_dict_1)
    # optimizer_1.load_state_dict(optimizer_dict_1)

    # model_dict_2, optimizer_dict_2, epoch_dict = checkpoint.load_checkpoint(
    #     os.path.join(checkpoint.save_path, 'checkpoint_030_model_2.pth'))
    # model_2.load_state_dict(model_dict_2)
    # optimizer_2.load_state_dict(optimizer_dict_2)

    # for i in range(epoch_dict):
    #     scheduler_1.step()
    # scheduler_2.step()

    time_start = time.time()

    best_epoch = 0
    best_valid_accuracy = 0
    best_peoch_train_accuracy = 0

    for epoch in range(epoch_dict, config.nEpochs + 1):
        cls_loss_, accuracy, accuracy_valid = trainer.train(epoch)
        if best_valid_accuracy < accuracy_valid:
            best_valid_accuracy = accuracy_valid
            best_peoch_train_accuracy = accuracy
            best_epoch = epoch

        if epoch == 5:
            checkpoint.save_checkpoint(model_1, optimizer_1, epoch=epoch, index=epoch, tag='model_1')
            checkpoint.save_checkpoint(model_2, optimizer_2, epoch=epoch, index=epoch, tag='model_2')
        if epoch == 10:
            checkpoint.save_checkpoint(model_1, optimizer_1, epoch=epoch, index=epoch, tag='model_1')
            checkpoint.save_checkpoint(model_2, optimizer_2, epoch=epoch, index=epoch, tag='model_2')
        if epoch == 15:
            checkpoint.save_checkpoint(model_1, optimizer_1, epoch=epoch, index=epoch, tag='model_1')
            checkpoint.save_checkpoint(model_2, optimizer_2, epoch=epoch, index=epoch, tag='model_2')

    time_end = time.time()
    print(time_end - time_start)
    print("\r\Best peoch: {}|===>Train Accuracy: {:.6f}   valid Accuracy: {:.6f}\r\n"
          .format(best_epoch, best_peoch_train_accuracy, best_valid_accuracy))

    # #计算样本量
    # imagedb.count_num_sample()
    #
    # #记录字符集
    # imagedb.write_char_set()
    #
    # #随机抽取部分字符为模型使用
    # imagedb.random_select()
    #
    # #加载数据
    # train_dataset, valid_dataset = imagedb.load_data()
    #
    # #Set dataloader
    # kwargs = {'num_workers': config.nThreads, 'pin_memory': True} if use_cuda else {}
    # train_loader = torch.utils.data.DataLoader(
    #     dataset=train_dataset,
    #     batch_size=config.batchSize,
    #     shuffle=True,
    #     **kwargs)
    #
    # valid_loader = torch.utils.data.DataLoader(
    #     dataset=valid_dataset,
    #     batch_size=config.batchSize,
    #     shuffle=True,
    #     **kwargs)
    #
    # #Set model
    # model =AlexNet()
    # model = model.to(device)
    #
    # # Set checkpoint
    # checkpoint = CheckPoint(config.save_path)
    #
    # # Set optimizer
    # optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=config.step, gamma=0.1)
    #
    # # Set trainer
    # logger = Logger(config.save_path)
    # trainer = AlexNetTrainer(config.lr, train_loader, valid_loader, model, optimizer, scheduler, logger, device)
    #
    # print(model)
    #
    # time_start = time.time()
    #
    # for epoch in range(1, config.nEpochs + 1):
    #     cls_loss_, accuracy, accuracy_valid = trainer.train(epoch)
    #     checkpoint.save_model(model, index=epoch)
    #
    # time_end = time.time()
    # print(time_end - time_start)
