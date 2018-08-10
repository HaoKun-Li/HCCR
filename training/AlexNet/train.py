import torch
import torchvision
from models.AlexNet import AlexNet
from training.AlexNet.trainer import AlexNetTrainer
from training.AlexNet.config import Config
from tools.logger import Logger
from checkpoint import CheckPoint
import os
import matplotlib.pyplot as plt
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

    #计算样本量
    imagedb.count_num_sample()

    #记录字符集
    imagedb.write_char_set()

    #随机抽取部分字符为模型使用
    imagedb.random_select()

    #加载数据
    train_dataset, valid_dataset = imagedb.load_data()

    #Set dataloader
    kwargs = {'num_workers': config.nThreads, 'pin_memory': True} if use_cuda else {}
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=config.batchSize,
        shuffle=True,
        **kwargs)

    valid_loader = torch.utils.data.DataLoader(
        dataset=valid_dataset,
        batch_size=config.batchSize,
        shuffle=True,
        **kwargs)

    #Set model
    model =AlexNet()
    model = model.to(device)

    # Set checkpoint
    checkpoint = CheckPoint(config.save_path)

    # Set optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=config.step, gamma=0.1)

    # Set trainer
    logger = Logger(config.save_path)
    trainer = AlexNetTrainer(config.lr, train_loader, valid_loader, model, optimizer, scheduler, logger, device)

    print(model)

    time_start = time.time()

    for epoch in range(1, config.nEpochs + 1):
        cls_loss_, accuracy, accuracy_valid = trainer.train(epoch)
        checkpoint.save_model(model, index=epoch)

    time_end = time.time()
    print(time_end - time_start)
