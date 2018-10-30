# -*- coding: utf-8 -*-
import torch
from torchvision import transforms
import numpy as np
import sys
from sklearn.cluster import KMeans
import os
sys.path.append('../../')

from models.FastCNN import FastCNN

from training.FastCNN.trainer import FastCNNTrainer
from training.FastCNN.config import Config
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


    os.environ['CUDA VISIBLE_DEVICES'] = config.GPU
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

    #记录字符集
    imagedb.write_char_set()



    # #Set dataloader
    kwargs = {'num_workers': config.nThreads, 'pin_memory': True} if use_cuda else {}
    transform_train = transforms.Compose([
        transforms.RandomCrop(size=(96, 96), padding=8),
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

    #Set model
    model = FastCNN()
    para = sum([np.prod(list(p.size())) for p in model.parameters()])
    print('Model {} : params: {:4f}M'.format(model._get_name(), para * 4 / 1000 / 1000))
    model = model.to(device)

    # Set checkpoint
    checkpoint = CheckPoint(config.save_path)

    # Set optimizer
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=config.lr)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=config.step, gamma=0.1)

    # Set trainer
    logger = Logger(config.save_path)
    trainer = FastCNNTrainer(config.lr, train_loader, valid_loader, model, optimizer, scheduler, logger, device)

    print(model)
    epoch_dict = 1

    # load_model
    model_dict, optimizer_dict, epoch_dict = checkpoint.load_checkpoint(
        os.path.join(checkpoint.save_path, 'checkpoint_090_model_1.pth'))
    model.load_state_dict(model_dict)
    optimizer.load_state_dict(optimizer_dict)

    for i in range(epoch_dict):
        scheduler.step()

    time_start = time.time()

    best_epoch = 0
    best_valid_accuracy = 0
    best_peoch_train_accuracy = 0

    # for epoch in range(1, config.nEpochs + 1):
    #     cls_loss_, accuracy, accuracy_valid = trainer.train(epoch)
    #     if best_valid_accuracy < accuracy_valid:
    #         best_valid_accuracy = accuracy_valid
    #         best_peoch_train_accuracy = accuracy
    #         best_epoch = epoch
    #
    #     if epoch == 90:
    #         checkpoint.save_checkpoint(model, optimizer, epoch=epoch, index=epoch, tag='model_1')
    #
    # time_end = time.time()
    # print(time_end - time_start)
    # print("\r\Best peoch: {}|===>Train Accuracy: {:.6f}   valid Accuracy: {:.6f}\r\n"
    #       .format(best_epoch, best_peoch_train_accuracy, best_valid_accuracy))

    featureMean = trainer.calculate_mean_feature().numpy()
    print(featureMean.shape)
    print(type(featureMean))

    time_start = time.time()
    kmeans = KMeans(n_clusters=2, random_state=0).fit(featureMean)
    time_end = time.time()
    print(time_end - time_start)

    np.save(config.save_path + '/kmeans_label.npy', kmeans.labels_)
    print(np.load(config.save_path + '/kmeans_label.npy'))