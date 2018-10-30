# -*- coding: utf-8 -*-

import cv2
import numpy as np
import torchvision
import matplotlib.pyplot as plt

print("test")

# arr = np.array([
#     [0, 0, 0, 0, 0],
#     [0, 0, 0, 0, 0],
#     [0, 1, 1, 1, 0],
#     [0, 1, 1, 1, 0],
#     [0, 0, 0, 0, 0],
# ], dtype='float32')
# print(type(arr[1][1]))
#
# fil_x = np.array(
#             [[-1, -2, -1],
#              [0, 0, 0],
#              [1, 2, 1]]
#         )
#
# print(type(fil_x[1][1]))
#
# res = cv2.filter2D(arr, -1, fil_x)
# print(res)
# print(type(res[1][1]))


def show_part_point_grid(grid, parts):

    plt.imshow(grid)
    plt.axis('off')
    image_size = 114
    for i in range(parts.shape[0]):
        for j in range(parts.shape[1]):
            plt.scatter(parts[i, j, 0] + i % 8 * (2+image_size),
                        parts[i, j, 1] + i // 8 * (2+image_size),
                        s=10, marker='.', c='r')

    plt.savefig('plt_image.jpg')

    return

def show_part_point_grid_boxs(grid, parts):

    img = cv2.cvtColor(grid*255, cv2.COLOR_RGB2BGR)
    image_size = 114
    l = 24
    colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (150, 150, 0)]
    for i in range(parts.shape[0]):
        for j in range(parts.shape[1]):
            box = np.array([np.maximum(0, (parts[i, j, 0] - l)), np.maximum(0, (parts[i, j, 1] - l)),
                                np.minimum(image_size, (parts[i, j, 0] + l)), np.minimum(image_size, (parts[i, j, 1] + l))])
            cv2.rectangle(img, (int(box[0] + i % 8 * (2+image_size)), int(box[1]) + i // 8 * (2+image_size)),
                          (int(box[2] + i % 8 * (2+image_size)), int(box[3]) + i // 8 * (2+image_size)), colors[j], 1)

    cv2.imwrite('plt_image_boxs.jpg', img)  # 保存图片

    return


# te = np.ones((64, 4, 2))
# np.save('parts.npy', te)
show_part_point_grid(np.load('grid.npy'), np.load('parts.npy'))
show_part_point_grid_boxs(np.load('grid.npy'), np.load('parts.npy'))

epoch = 1
batch_idx = 10
if epoch == 1 or epoch == 100 and batch_idx == 1:
    print('well')