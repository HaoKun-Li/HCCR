# -*- coding: utf-8 -*-

import os
import numpy as np
import random
import torch
import cv2

print("test")

arr = np.array([
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0],
    [0, 1, 1, 1, 0],
    [0, 1, 1, 1, 0],
    [0, 0, 0, 0, 0],
], dtype='float32')
print(type(arr[1][1]))

fil_x = np.array(
            [[-1, -2, -1],
             [0, 0, 0],
             [1, 2, 1]]
        )

print(type(fil_x[1][1]))

res = cv2.filter2D(arr, -1, fil_x)
print(res)
print(type(res[1][1]))