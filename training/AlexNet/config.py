# -*- coding: utf-8 -*-
import os

class Config(object):
    def __init__(self):
        super(Config, self).__init__()
        #  ------------ General options ----------------------------------------
        self.save_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))+"/results/AlexNet"
        self.dataPath = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))+"/data/"
        self.trainDataPath = self.dataPath+"/HWDB1.1trn_gnt"
        self.validDataPath = self.dataPath + "/HWDB1.1tst_gnt"
        # self.annoPath = "./annotations/imglist_anno_12.txt"
        self.manualSeed = 1  # manually set RNG seed
        self.use_cuda = True
        self.GPU = "3"  # default gpu to use

        # ------------- Data options -------------------------------------------
        self.nThreads = 2  # number of data loader threads
        self.random_size = 3755 #number of random select Chinese
        self.resize_size = 114

        # ---------- Optimization options --------------------------------------
        self.nEpochs = 1000  # number of total epochs to train 400
        self.batchSize = 32  # mini-batch size 128

        # lr master for optimizer 1 (mask vector d)
        self.lr = 0.01  # initial learning rate
        self.step = [10, 25, 40]  # step for linear or exp learning rate policy
        self.decayRate = 0.5  # lr decay rate
        self.endlr = -1

        # ---------- Model options ---------------------------------------------
        self.experimentID = "072402"

        # ---------- Resume or Retrain options ---------------------------------------------
        self.resume = None  # "./checkpoint_064.pth"
        self.retrain = None

        self.save_path = self.save_path + "log_bs{:d}_lr{:.3f}_{}/".format(self.batchSize, self.lr, self.experimentID)

