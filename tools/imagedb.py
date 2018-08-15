# -*- coding: utf-8 -*-

import os
import numpy as np
import scipy.misc
import struct
import PIL.Image
import random
from training.AlexNet.config import Config
import torch
import cv2
import time
import torch.utils.data as data



def png_loader(path):
    with open(path, 'rb') as f:
        img = PIL.Image.open(f)
        return img.convert('L')

class HCDataset(data.Dataset):
    def __init__(self, image_annotation_file, prefix_path='', transform=None, is_train=False):
        self.image_annotation_file = image_annotation_file
        self.prefix_path = prefix_path
        self.is_train = is_train
        self.image_set_index = self.load_image_set_index()
        self.num_images = len(self.image_set_index)
        self.gt_imdb = self.load_annotations()
        # if self.is_train:
        self.transform = transform
        self.loader = png_loader


    def load_image_set_index(self):
        """Get image index

                Parameters:
                ----------
                Returns:
                -------
                image_set_index: str
                    relative path of image
                """
        assert os.path.exists(self.image_annotation_file), 'Path does not exist: {}'.format(
            self.image_annotation_file)
        with open(self.image_annotation_file, 'r') as f:
            image_set_index = [x.strip().split(' ')[0] for x in f.readlines()]
        return image_set_index


    def real_image_path(self, index):
        """Given image index, return full path

        Parameters:
        ----------
        index: str
            relative path of image
        Returns:
        -------
        image_file: str
            full path of image
        """

        index = index.replace("\\", "/")

        if not os.path.exists(index):
            image_file = os.path.join(self.prefix_path, index)
        else:
            image_file = index
        if not image_file.endswith('.png'):
            image_file = image_file + '.png'
        assert os.path.exists(
            image_file), 'Path does not exist: {}'.format(image_file)
        return image_file


    def load_annotations(self):
        """Load annotations

        Returns:
        -------
        imdb: dict
            image database with annotations
        """

        assert os.path.exists(self.image_annotation_file), 'annotations not found at {}'.format(
            self.image_annotation_file)
        with open(self.image_annotation_file, 'r') as f:
            annotations = f.readlines()

        with open(config.dataPath + 'char_set', 'r') as f:
            char_set = f.read().strip().split(' ')
            print("read char_set")

        imdb = []
        for i in range(self.num_images):
            annotation = annotations[i].strip().split(' ')
            index = annotation[0]
            im_path = self.real_image_path(index)
            imdb_ = dict()

            imdb_['image'] = im_path
            imdb_['label'] = char_set.index(annotation[1])
            imdb.append(imdb_)
        return imdb


    def __len__(self):
        return self.num_images


    def __getitem__(self, idx):
        imdb_ = self.gt_imdb[idx]
        image = self.loader(imdb_['image'])
        label = imdb_['label']

        if self.transform:
            image = self.transform(image)
            image = image.type(torch.float32)

        return image, label









config = Config()


# convert to png and record path
def preprocess_gnt():
    train_png_path = os.path.join(config.dataPath, 'train_png')
    if not os.path.isdir(train_png_path):
        os.makedirs(train_png_path)

    valid_png_path = os.path.join(config.dataPath, 'valid_png')
    if not os.path.isdir(valid_png_path):
        os.makedirs(valid_png_path)

    train_annotation = []
    train_annotation_path = os.path.join(config.dataPath, 'train_png_annotation.txt')

    if not os.path.exists(train_annotation_path):
        for image, tagcode, file_name in read_from_gnt_dir(config.trainDataPath):
            tagcode_unicode = struct.pack('>H', tagcode).decode('gb2312')
            image = resize_and_normalize_image(image)
            im = PIL.Image.fromarray(image)
            im_path = os.path.join(train_png_path, file_name + '_' + str(tagcode) + '.png')
            im.convert('L').save(im_path)
            train_annotation.append(os.path.split(im_path)[1]+' '+str(tagcode))
            print(im_path)
            png_loader(im_path)


        with open(train_annotation_path, 'w') as f:
            for line in train_annotation:
                f.write(line+'\n')

    valid_annotation = []
    valid_annotation_path = os.path.join(config.dataPath, 'valid_png_annotation.txt')
    if not os.path.exists(valid_annotation_path):
        for image, tagcode, file_name in read_from_gnt_dir(config.validDataPath):
            tagcode_unicode = struct.pack('>H', tagcode).decode('gb2312')
            image = resize_and_normalize_image(image)
            im = PIL.Image.fromarray(image)
            im_path = os.path.join(valid_png_path, file_name + '_' + str(tagcode) + '.png')
            im.convert('RGB').save(im_path)
            valid_annotation.append(os.path.split(im_path)[1]+' '+str(tagcode))
            print(im_path)

        with open(valid_annotation_path, 'w') as f:
            for line in valid_annotation:
                f.write(line+'\n')

# 读取图像和对应的汉字
def read_from_gnt_dir(gnt_dir):
    def one_file(f):
        header_size = 10
        while True:
            header = np.fromfile(f, dtype='uint8', count=header_size)
            if not header.size:
                break
            sample_size = header[0] + (header[1]<<8) + (header[2]<<16) + (header[3]<<24)
            tagcode = header[5] + (header[4]<<8)
            width = header[6] + (header[7] << 8)
            height = header[8] + (header[9] << 8)
            if header_size + width*height != sample_size:
                break
            image = np.fromfile(f, dtype='uint8', count=width*height).reshape((height, width))
            yield image, tagcode



    for file_name in os.listdir(gnt_dir):
        if file_name.endswith('.gnt'):
            file_path = os.path.join(gnt_dir, file_name)
            file_name = os.path.split(file_name)[1]
            file_name = os.path.splitext(file_name)[0]
            with open(file_path, 'rb') as f:
                for image, tagcode in one_file(f):
                    yield image, tagcode, file_name


#梯度特征提取
def sobel_1(image):
    gradient = np.zeros(image.shape, dtype='float32')
    for i in range(1, image.shape[0]-1):
        for j in range(1, image.shape[1]-1):
            gradient[i][j] = image[i-1][j-1]*(-1) + image[i-1][j+1]*1 + image[i][j-1]*(-2) + image[i][j+1] * 2 +\
                             image[i+1][j-1]*(-1) + image[i+1][j+1]*1

    return gradient


def filter_1(image):
    fil = np.array(
        [[-1, 0, 1],
         [-2, 0, 2],
         [-1, 0, 1]]
    )
    res = cv2.filter2D(image, -1, fil)

    res = (res - np.min(res)) / (np.max(res) - np.min(res)) * 255
    return res

def filter_2(image):
    fil = np.array(
        [[-2, -1, 0],
         [-1, 0, 1],
         [0, 1, 2]]
    )
    res = cv2.filter2D(image, -1, fil)

    res = (res - np.min(res)) / (np.max(res) - np.min(res)) * 255
    return res

def filter_3(image):
    fil = np.array(
        [[-1, -2, -1],
         [0, 0, 0],
         [1, 2, 1]]
    )
    res = cv2.filter2D(image, -1, fil)

    res = (res - np.min(res)) / (np.max(res) - np.min(res)) * 255
    return res

def filter_4(image):
    fil = np.array(
        [[0, -1, -2],
         [1, 0, -1],
         [2, 1, 0]]
    )
    res = cv2.filter2D(image, -1, fil)

    res = (res - np.min(res)) / (np.max(res) - np.min(res)) * 255
    return res

def filter_5(image):
    fil = np.array(
        [[1, 0, -1],
         [2, 0, -2],
         [1, 0, -1]]
    )
    res = cv2.filter2D(image, -1, fil)

    res = (res - np.min(res)) / (np.max(res) - np.min(res)) * 255
    return res

def filter_6(image):
    fil = np.array(
        [[2, 1, 0],
         [1, 0, -1],
         [0, -1, -2]]
    )
    res = cv2.filter2D(image, -1, fil)

    res = (res - np.min(res)) / (np.max(res) - np.min(res)) * 255
    return res

def filter_7(image):
    fil = np.array(
        [[1, 2, 1],
         [0, 0, 0],
         [-1, -2, -1]]
    )
    res = cv2.filter2D(image, -1, fil)

    res = (res - np.min(res)) / (np.max(res) - np.min(res)) * 255
    return res

def filter_8(image):
    fil = np.array(
        [[0, 1, 2],
         [-1, 0, 1],
         [-2, -1, 0]]
    )
    res = cv2.filter2D(image, -1, fil)

    res = (res - np.min(res)) / (np.max(res) - np.min(res)) * 255
    return res



 # 统计样本数
def count_num_sample():
    train_counter = 0
    valid_counter = 0

    if not os.path.isdir(config.save_path + '/png/'):
        os.makedirs(config.save_path + '/png/')

    for image, tagcode in read_from_gnt_dir(config.trainDataPath):
        tagcode_unicode = struct.pack('>H', tagcode).decode('gb2312')
        train_counter += 1

        # 提取部分图像，转为png
        st = time.time()
        if train_counter < 10:
            im = PIL.Image.fromarray(image)
            im.convert('RGB').save(config.save_path + '/png/' + tagcode_unicode + str(train_counter) + '.png')

            image = resize_and_normalize_image(image)
            im = PIL.Image.fromarray(image)
            im.convert('RGB').save(config.save_path + '/png/' + tagcode_unicode + str(train_counter) + '_resize.png')

    for image, tagcode in read_from_gnt_dir(config.validDataPath):
        tagcode_unicode = struct.pack('>H', tagcode).decode('gb2312')
        valid_counter += 1

    # 样本数
    print('train_counter: '+str(train_counter), 'valid_counter: '+str(valid_counter))


#resize to 64*64
def resize_and_normalize_image(img):
    # 补方
    pad_size = abs(img.shape[0] - img.shape[1]) // 2
    if img.shape[0] < img.shape[1]:
        pad_dims = ((pad_size, pad_size), (0, 0))
    else:
        pad_dims = ((0, 0), (pad_size, pad_size))

    img = np.lib.pad(img, pad_dims, mode='constant', constant_values=255)

    # im = PIL.Image.fromarray(img)
    # im.convert('RGB').save('png/' + str(train_counter) + tagcode_unicode + '补方+pad_size_' + str(pad_size) + '.png')

    # 缩放
    img = scipy.misc.imresize(img, (config.resize_size-6, config.resize_size-6))
    img = np.lib.pad(img, ((3, 3), (3, 3)), mode='constant', constant_values=255)
    assert img.shape == (config.resize_size, config.resize_size)

    # img = img.flatten()

    # 像素值范围0到1(Min-max normalization),最亮的设为0， 最暗设为255
    img = (1 - (img - np.min(img)) / (np.max(img) - np.min(img))) * 255
    img = img.astype(np.float32)
    return img


def gradient_feature_maps(image):

    image_1 = filter_1(image)
    image_2 = filter_2(image)
    image_3 = filter_3(image)
    image_4 = filter_4(image)
    image_5 = filter_5(image)
    image_6 = filter_6(image)
    image_7 = filter_7(image)
    image_8 = filter_8(image)
    images = np.array([image, image_1, image_2, image_3, image_4, image_5, image_6, image_7, image_8])
    return images


def write_char_set():
    if not os.path.exists(config.dataPath+'char_set'):
        char_set = []
        for image, tagcode, _ in read_from_gnt_dir(gnt_dir=config.trainDataPath):
            # tagcode_unicode = struct.pack('>H', tagcode).decode('gb2312')
            if tagcode not in char_set:
                char_set.append(tagcode)

        with open(config.dataPath+'char_set', 'w') as f:
            for line in char_set:
                f.write(str(line)+' ')
            print("write char_set")

        print('char_set_length: '+str(len(char_set)))


# 生成随机汉字列表
def random_select():
    if not os.path.exists(config.save_path + 'random_char_set'):
        with open(config.save_path+'char_set', 'r') as f:
            char_set = f.read()
            print("read char_set")

        # 生成随机数列表
        set = list(range(1, len(char_set) + 1))
        randomNum = random.sample(set, config.random_size)

        # 生成随机汉字列表
        random_set = ''
        for i in randomNum:
            random_set += char_set[i - 1]

        with open(config.save_path + 'random_char_set', 'w') as f:
            f.write(random_set)
            print("write random_char_set")

    else:
        with open(config.save_path+'random_char_set', 'r') as f:
            random_set = f.read()
            print("read random_char_set")

    if len(random_set)!=config.random_size:
        os.remove(config.save_path+'random_char_set')
        random_select()

    else:
        print('random_set: ' + random_set)
        print('len(random_set): ' + str(len(random_set)))


#加载数据
def load_data():
    def convert_to_one_hot(char):
        vector = np.zeros(len(random_set))
        vector[random_set.index(char)] = 1
        return vector


    with open(config.save_path + 'random_char_set', 'r') as f:
        random_set = f.read()
        print("read random_char_set")

    train_data_x = []
    train_data_y = []
    for image, tagcode in read_from_gnt_dir(gnt_dir=config.trainDataPath):
        tagcode_unicode = struct.pack('>H', tagcode).decode('gb2312')
        if tagcode_unicode in random_set:
            train_data_x.append(resize_and_normalize_image(image))
            train_data_y.append(random_set.index(tagcode_unicode))
            # train_data_y.append(convert_to_one_hot(tagcode_unicode))

    print('len(train_data_x): ' + str(len(train_data_x)))

    valid_data_x = []
    valid_data_y = []

    for image, tagcode in read_from_gnt_dir(gnt_dir=config.validDataPath):
        tagcode_unicode = struct.pack('>H', tagcode).decode('gb2312')
        if tagcode_unicode in random_set:
            valid_data_x.append(resize_and_normalize_image(image))
            valid_data_y.append(random_set.index(tagcode_unicode))
            # valid_data_y.append(convert_to_one_hot(tagcode_unicode))

    print('len(valid_data_x): ' + str(len(valid_data_x)))

    # 转为torch.FloatTensor
    train_data_x = torch.from_numpy(np.array(train_data_x)).type(torch.FloatTensor)
    train_data_y = torch.from_numpy(np.array(train_data_y)).type(torch.LongTensor)

    valid_data_x = torch.from_numpy(np.array(valid_data_x)).type(torch.FloatTensor)
    valid_data_y = torch.from_numpy(np.array(valid_data_y)).type(torch.LongTensor)

    # 改变输入的张量形状
    train_data_x = train_data_x.view(-1, 1, config.resize_size, config.resize_size)
    valid_data_x = valid_data_x.view(-1, 1, config.resize_size, config.resize_size)

    # 生成dataset
    train_dataset = torch.utils.data.TensorDataset(train_data_x, train_data_y)
    valid_dataset = torch.utils.data.TensorDataset(valid_data_x, valid_data_y)
    return train_dataset, valid_dataset