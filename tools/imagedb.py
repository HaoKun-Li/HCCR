# -*- coding: utf-8 -*-

import os
import numpy as np
import scipy.misc
import struct
from PIL import Image
import random
from training.AlexNet.config import Config
import torch
import cv2
import time
import torch.utils.data as data



def png_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
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
            image = self.gradient_feature_maps(image.numpy(), label)
            image = torch.from_numpy(image)

        return image, label

    def filter_x(self, image):
        fil = np.array(
            [[-1, 0, 1],
             [-2, 0, 2],
             [-1, 0, 1]]
        )
        res = cv2.filter2D(image, -1, fil)/8
        print(res[0][30][40:50])
        res = (res - np.min(res)) / (np.max(res) - np.min(res)) * 255
        print(res[0][30][40:50])
        return res

    def filter_y(self, image):
        fil = np.array(
            [[-1, -2, -1],
             [0, 0, 0],
             [1, 2, 1]]
        )
        res = cv2.filter2D(image, -1, fil)/8
        res = (res - np.min(res)) / (np.max(res) - np.min(res)) * 255
        return res


    def save_as_png(self, image, label, st):
        image = (image - np.min(image)) / (np.max(image) - np.min(image)) * 255

        im = Image.fromarray(image)
        im.convert('RGB').save(config.save_path + '/gradient_png/' + str(label) + '_' + str(st) + '.png')



    def gradient_feature_maps(self, image, label):
        fil_x = np.array(
            [[-1, 0, 1],
             [-2, 0, 2],
             [-1, 0, 1]], dtype='float32'
        )
        fil_y = np.array(
            [[-1, -2, -1],
             [0, 0, 0],
             [1, 2, 1]], dtype='float32'
        )

        image = image[0]
        res_x = cv2.filter2D(image, -1, fil_x)/4
        res_y = cv2.filter2D(image, -1, fil_y)/4
        res_3 = -1 * res_x
        res_4 = -1 * res_y
        res_5 = 0.707 * res_x + 0.707 * res_y
        res_6 = 0.707 * res_x - 0.707 * res_y
        res_7 = -0.707 * res_x + 0.707 * res_y
        res_8 = -0.707 * res_x - 0.707 * res_y

        # if not os.path.isdir(config.save_path + '/gradient_png/'):
        #     os.makedirs(config.save_path + '/gradient_png/')
        # self.save_as_png(image.reshape(config.resize_size, config.resize_size), label, 'origin')
        # self.save_as_png(res_x.reshape(config.resize_size, config.resize_size), label, 'x')
        # self.save_as_png(res_y.reshape(config.resize_size, config.resize_size), label, 'y')
        # self.save_as_png(res_3.reshape(config.resize_size, config.resize_size), label, 3)
        # self.save_as_png(res_4.reshape(config.resize_size, config.resize_size), label, 4)
        # self.save_as_png(res_5.reshape(config.resize_size, config.resize_size), label, 5)
        # self.save_as_png(res_6.reshape(config.resize_size, config.resize_size), label, 6)
        # self.save_as_png(res_7.reshape(config.resize_size, config.resize_size), label, 7)
        # self.save_as_png(res_8.reshape(config.resize_size, config.resize_size), label, 8)

        # images = np.array([image, res_x, res_y, res_3, res_4, res_5, res_6, res_7, res_8]).reshape(-1, config.resize_size, config.resize_size)
        images = np.array([image]).reshape(-1, config.resize_size, config.resize_size)
        return images









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

            #test
            # im = Image.fromarray(image)
            # im_path = os.path.join(train_png_path, file_name + '_orgin' + str(tagcode) + '.png')
            # im.convert('L').save(im_path)
            #test

            image = forward_nonlinear_1d(src = image, dst_wid = config.resize_size-6, dst_hei = config.resize_size-6, ratio_preserve_func = 'SQUART')
            image = np.lib.pad(image, ((3, 3), (3, 3)), mode='constant', constant_values=0)
            assert image.shape == (config.resize_size, config.resize_size)
            image = image.astype(np.float32)

            im = Image.fromarray(image)
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
            image = forward_nonlinear_1d(src = image, dst_wid = config.resize_size-6, dst_hei = config.resize_size-6, ratio_preserve_func = 'SQUART')
            image = np.lib.pad(image, ((3, 3), (3, 3)), mode='constant', constant_values=0)
            assert image.shape == (config.resize_size, config.resize_size)
            image = image.astype(np.float32)
            
	    # image = resize_and_normalize_image(image)
            
	    im = Image.fromarray(image)
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
            im = Image.fromarray(image)
            im.convert('RGB').save(config.save_path + '/png/' + tagcode_unicode + str(train_counter) + '.png')

            image = resize_and_normalize_image(image)
            im = Image.fromarray(image)
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

    # img = Image.fromarray(img)
    # img = img.resize((config.resize_size - 6, config.resize_size - 6))
    # img = np.asarray(img)

    img = np.lib.pad(img, ((3, 3), (3, 3)), mode='constant', constant_values=255)
    assert img.shape == (config.resize_size, config.resize_size)

    # img = img.flatten()

    # 像素值范围0到1(Min-max normalization),最亮的设为0， 最暗设为255
    img = (1 - (img - np.min(img)) / (np.max(img) - np.min(img))) * 255
    img = img.astype(np.float32)
    return img





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


def aspect_radio_mapping(r1, dst_wid, dst_hei, ratio_preserve_func):
    if ratio_preserve_func == 'ASPECT':
        return r1

    elif ratio_preserve_func == 'SQUART':
        return np.sqrt(r1)

    elif ratio_preserve_func == 'CUBIC':
        return np.power(r1, 0.333)

    elif ratio_preserve_func == 'SINE':
        return np.sqrt(np.sin(3.1415926*r1/2))

    else:
        return min(dst_wid, dst_hei) / max(dst_wid, dst_hei)


def forward_push_val(dst, dst_wid, dst_hei, val, x, y, xscale, yscale):
    fl = x - xscale / 2
    fr = x + xscale / 2
    ft = y - yscale / 2
    fb = y + yscale / 2

    l = int(fl)
    r = int(fr) + 1
    t = int(ft)
    b = int(fb) + 1

    l = min(max(l, 0), dst_wid - 1)
    r = min(max(r, 0), dst_wid - 1)
    t = min(max(t, 0), dst_hei - 1)
    b = min(max(b, 0), dst_hei - 1)

    for j in range(t, b+1):
        for i in range(l, r+1):
            # float intersect_area;
            xg = min(i+1, fr) - max(i, fl)
            yg = min(j+1, fb) - max(j, ft)

            if xg > 0 and yg > 0:
                dst[j, i] += xg * yg * val

    return dst


def forward_push_val2(dst, dst_wid, dst_hei, val, fl, ft, fr, fb, xscale, yscale):
    l = int(fl)
    r = int(fr) + 1
    t = int(ft)
    b = int(fb) + 1

    l = min(max(l, 0), dst_wid - 1)
    r = min(max(r, 0), dst_wid - 1)
    t = min(max(t, 0), dst_hei - 1)
    b = min(max(b, 0), dst_hei - 1)

    for j in range(t, b):
        for i in range(l, r):
            xg = min(i + 1, fr) - max(i, fl)
            yg = min(j + 1, fb) - max(j, ft)

            if xg > 0 and yg > 0:
                dst[j, i] += xg * yg * val

    return dst


def forward_nonlinear_1d(src, dst_wid, dst_hei, ratio_preserve_func):

    src_wid = src.shape[1]
    src_hei = src.shape[0]
    region = [0, 0, src_wid-1, src_hei-1]
    m10 = 0
    m01 = 0
    u20 = 0
    u02 = 0
    constval = 0.001
    threshold = 64 # smaller than 128 are considered as background pixel while others are foreground pixels.
    src = (1 - (src - np.min(src)) / (np.max(src) - np.min(src))) * 255
    dst = np.zeros([dst_hei, dst_wid])

    for y in range(region[1], region[3]+1):
        for x in range(region[0], region[2]+1):
            m10 += x * src[y][x]
            m01 += y * src[y][x]

    m00 = np.sum(src)
    if m00 == 0:
        return;

    #xc, yc
    xc = m10 / m00
    yc = m01 / m00

    # general u20, u02
    for y in range(region[1], region[3]+1):
        for x in range(region[0], region[2]+1):
            u20 += (x - xc) * (x - xc) * src[y][x]
            u02 += (y - yc) * (y - yc) * src[y][x]

    #general w1, h1
    w1 = int(np.round(4.5 * np.sqrt(u20 / m00)))
    h1 = int(np.round(4.5 * np.sqrt(u02 / m00)))

    l = np.round(xc - w1 / 2)
    r = np.round(xc + w1 / 2 + 1)
    t = np.round(yc - h1 / 2)
    b = np.round(yc + h1 / 2 + 1)
    l = int(min(max(l, 0), src_wid))
    r = int(min(max(r, 0), src_wid))
    t = int(min(max(t, 0), src_hei))
    b = int(min(max(b, 0), src_hei))

    dx = np.zeros([(b - t), (r - l)])
    dy = np.zeros([(b - t), (r - l)])
    px = np.zeros([(r - l), 1])
    py = np.zeros([(b - t), 1])
    hx = np.zeros([(r - l), 1])
    hy = np.zeros([(b - t), 1])

    #general dx
    for y in range(t, b):
        run_start = -1
        run_end = -1
        for x in range(l, r):
            if src[y][x] < threshold:
                if run_start < 0:
                    run_start = x
                    run_end = x
                else:
                    run_end = x
            else:
                if run_start < 0:
                    dx[y-t][x-l] = constval
                else:
                    d = 1. / (w1/4 + run_end - run_start +1)
                    dx[y - t][x - l] = constval
                    for i in range(run_start, run_end+1):
                        dx[y - t][i - l] = d
                    run_end = -1
                    run_start = -1

        if run_start > 0:
            d = 1. / (w1/4 + run_end - run_start + 1)
            for i in range(run_start, run_end + 1):
                dx[y - t][i - l] = d

    # general dy
    for x in range(l, r):
        run_start = -1
        run_end = -1
        for y in range(t, b):
            if src[y][x] < threshold:
                if run_start < 0:
                    run_start = y
                    run_end = y
                else:
                    run_end = y
            else:
                if run_start < 0:
                    dy[y-t][x-l] = constval
                else:
                    d = 1. / (h1/4 + run_end - run_start +1)
                    dy[y - t][x - l] = constval
                    for i in range(run_start, run_end+1):
                        dy[i - t][x - l] = d
                    run_end = -1
                    run_start = -1

        if run_start > 0:
            d = 1. / (h1/4 + run_end - run_start + 1)
            for i in range(run_start, run_end + 1):
                dy[i - t][x - l] = d

    #general dx_sum, dy_sum
    dx_sum = np.sum(dx)
    dy_sum = np.sum(dy)

    # general px, py
    py = (np.sum(dy, 1) / dy_sum)
    px = (np.sum(dx, 0) / dx_sum)

    #hx, hy
    for x in range(l, r):
        for i in range(l, x):
            hx[x-l] += px[i-l]

    for y in range(t, b):
        for j in range(t, y):
            hy[y-t] +=py[j-t]

    r1 = min((r-l), (b-t)) / max((r-l), (b-t))
    r2 = aspect_radio_mapping(r1, dst_wid, dst_hei, ratio_preserve_func)

    if w1 > h1:
        w2 = dst_wid
        h2 = int(w2 * r2)
        xoffset = 0
        yoffset = (dst_hei - h2) / 2

    else:
        h2 = dst_hei
        w2 = int(h2*r2)
        xoffset = (dst_wid - w2) / 2
        yoffset = 0

    xscale = w2 / w1
    yscale = h2 / h1

    # forward mapping
    for y in range(t, b):
        for x in range(l, r):
            x1 = w2 * hx[x-l]
            y1 = h2 * hy[y-t]
            # if src[y][x] > threshold:
            #     src[y][x] = src[y][x] # todo

            if y == b-1 or x == r-1:
                dst = forward_push_val(dst, dst_wid, dst_hei, src[y][x], x1 + xoffset, y1 + yoffset, xscale, yscale)

            else:
                x2 = w2 * hx[x - l + 1]
                y2 = h2 * hy[y - t + 1]
                dst = forward_push_val2(dst, dst_wid, dst_hei, src[y][x], x1 + xoffset, y1 + yoffset, x2 + xoffset, y2 + yoffset, xscale, yscale)

    return dst
