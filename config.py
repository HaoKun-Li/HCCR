import os


MODEL_STORE_DIR = './models'

TRAIN_DATA_DIR = './data'

#获取当前运行脚本的绝对路径（去掉最后2个路径）
LOG_DIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))+'/log'

USE_CUDA = True

TRAIN_DATA_DIR = 64

TRAIN_LR = 0.001

END_EPOCH = 2

