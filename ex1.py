# -*- coding: utf-8 -*-
# @Time    : 2023/3/19 19:33
# @Author  : Lufeng Wang
# @WeChat  : tofind404
# @File    : ex1.py
# @Software: PyCharm
import random

import numpy as np

import torch
from torchvision import transforms

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from PIL import Image

from torchtoolbox.transform import Cutout


def show_array(array_img, name):
    """
    矩阵， 图像
    :param array_img:
    :param name:
    :return:
    """
    im = plt.imshow(array_img, cmap='Pastel1')
    values = np.unique(array_img.ravel())
    # get the colors of the values, according to the
    # colormap used by imshow
    colors = [im.cmap(im.norm(value)) for value in values]
    # create a patch (proxy artist) for every color
    patches = [mpatches.Patch(color=colors[i], label="Mask {l}".format(l=int(values[i])))
               for i in range(len(values))]
    # put those patched as legend-handles into the legend
    plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., fontsize=7)  # 24

    # plt.grid(True)
    plt.title(name, fontsize=10)  # 20


#
# array_random = torch.randint(0, 2, (4, 256, 256))
#
# print(array_random)
#
# show_array(array_random[1,:,:], 'debut')
# plt.show()
#
# train_transformer = transforms.RandomCrop(256, padding=44, pad_if_needed=True, fill=0)
# array_transformer = train_transformer(array_random)
# show_array(array_transformer[1,:,:], 'transformer')
# plt.show()
#
# array_random =Image.open(r'C:\Users\Administrator\Desktop\singleBuilding_unet_graph\12.jpg')
# train_transformer2 = Cutout()
# array_transformer3 = train_transformer2(array_random)
# # show_array(array_transformer, 'transformer2')
# # plt.show()
# array_transformer3.show()

# def fill_array(base_array, shape_to_be_inserted):
#     row_loc = np.random.randint(low=0, high=(base_array.shape[0] - shape_to_be_inserted.shape[0] + 1))
#     col_loc = np.random.randint(low=0, high=(base_array.shape[1] - shape_to_be_inserted.shape[1] + 1))
#     base_array[row_loc:row_loc + shape_to_be_inserted.shape[0],
#     col_loc:col_loc + shape_to_be_inserted.shape[1]] = shape_to_be_inserted
#     return base_array
#
#


#
# import numpy as np
#
#
# def make_mask(array_img, list_mask, list_channel_fill=[0, 4, 0, 0]):
#     # convert list_mask to a numpy array of shape (n, 2)
#     array_mask = np.array(list_mask)
#     # get the row and column indices from the array_mask
#     row_indices = array_mask[:, 0]
#     col_indices = array_mask[:, 1]
#     # create a boolean mask of shape (height, width) where True indicates the pixels to be filled
#     bool_mask = np.zeros_like(array_img[0], dtype=bool)
#     bool_mask[row_indices, col_indices] = True
#     # create a fill array of shape (channels,) with the values from list_channel_fill
#     fill_array = np.array(list_channel_fill)
#     # use broadcasting and indexing to assign the fill values to the masked pixels in each channel
#     array_img[:, bool_mask] = fill_array[:, None]
#
#     return array_img


array_random = np.random.randint(5, 7, (4, 256, 256))
show_array(array_random[0, :, :], '0')
plt.show()

list_points = mask_rectangle_Cut_out(256, 0.5, 0.5)
plt.scatter(np.array(list_points).reshape(-1, 2)[:, 0], np.array(list_points).reshape(-1, 2)[:, 1])
plt.show()

array_random = make_mask(array_random, list_points, list_channel_fill=[0, 4, 0, 0])
for i in range(0, 4):
    show_array(array_random[i, :, :], str(i))
    plt.show()
