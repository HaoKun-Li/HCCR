# -*- coding: utf-8 -*-
import torch
import sys
import os
sys.path.append('../../')
import torchvision
from models.LeNet_5 import LeNet_5
from training.LeNet_5.trainer import LeNet_5Trainer
from training.LeNet_5.config import Config
from tools.logger import Logger
from checkpoint import CheckPoint
# import matplotlib.pyplot as plt
import multiprocessing

if __name__ == '__main__':
    multiprocessing.freeze_support()
    config = Config()
    if not os.path.exists(config.save_path):
        os.makedirs(config.save_path)


    os.environ['CUDA VISIBLE_DEVICES'] = config.GPU
    use_cuda = config.use_cuda and torch.cuda.is_available()
    print('use cuda'+str(use_cuda))
    torch.manual_seed(config.manualSeed)
    torch.cuda.manual_seed(config.manualSeed)
    device = torch.device("cuda" if use_cuda else "cpu")
    torch.backends.cudnn.benchmark = True

    #download train data
    transf = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor()
    ])

    train_data = torchvision.datasets.MNIST(
            root=config.dataPath,
            train=True,
            transform=transf,    # Converts a PIL.Image or numpy.ndarray to
                                                        #  torch.FloatTensor of shape (C x H x W) and normalize in the range [0.0, 1.0]
            download=config.download_mnist
        )

    print(train_data.train_data.size())
    print(train_data.train_labels.size())
    # plt.imshow(train_data.train_data[0].numpy(), cmap='gray')
    # plt.title('%i' % train_data.train_labels[0])
    # plt.show()


    #set dataloader
    kwargs = {'num_workers': config.nThreads, 'pin_memory': True} if use_cuda else {}
    train_loader = torch.utils.data.DataLoader(
        dataset = train_data,
        batch_size=config.batchSize,
        shuffle=True,
        **kwargs
    )

    #Set valid_data
    valid_data = torchvision.datasets.MNIST(
        root=config.dataPath,
        train=False
    )
    valid_x = torch.unsqueeze(valid_data.test_data, dim=1).type(torch.FloatTensor)[:2000]/255.    # shape from (2000, 28, 28) to (2000, 1, 28, 28), value in range(0,1)
    valid_y = valid_data.test_labels[:2000]


    #Set model
    model = LeNet_5()
    model = model.to(device)

    #Set checkpoint
    checkpoint = CheckPoint(config.save_path)

    #Set optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    scheduler = None

    #Set trainer
    logger = Logger(config.save_path)
    trainer = LeNet_5Trainer(config.lr, train_loader, valid_x, valid_y, model, optimizer, scheduler, logger, device)

    epoch_dict = 1
    model_dict, optimizer_dict, epoch_dict= checkpoint.load_checkpoint(os.path.join(checkpoint.save_path, 'checkpoint_005.pth'))
    model.load_state_dict(model_dict)
    optimizer.load_state_dict(optimizer_dict)


    for epoch in range(epoch_dict, config.nEpochs + 1):
        cls_loss_, accuracy= trainer.train(epoch)
        checkpoint.save_checkpoint(model, optimizer, epoch=epoch, index=epoch, tag="123123")


