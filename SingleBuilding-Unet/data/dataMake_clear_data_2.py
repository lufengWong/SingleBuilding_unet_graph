import pickle
import shutil

import numpy as np
from PIL import Image
import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from tqdm import tqdm


# 　删除重复的png

def show_array(array_img, name):
    """
    矩阵， 图像
    :param array_img:
    :param name:
    :return:
    """
    im = plt.imshow(array_img, cmap='rainbow')

    values = np.unique(array_img.ravel())
    # get the colors of the values, according to the
    # colormap used by imshow
    colors = [im.cmap(im.norm(value)) for value in values]
    # create a patch (proxy artist) for every color
    patches = [mpatches.Patch(color=colors[i], label="Label {l}".format(l=int(values[i]))) for i in range(len(values))]
    # put those patched as legend-handles into the legend
    plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

    # plt.grid(True)
    plt.title(name)


def similar(array_a, array_b):
    res = np.array(array_a == array_b)
    count_ture = np.sum(res == True)
    print(count_ture)
    num_ = array_a.shape[0] * array_a.shape[1]
    print(num_)
    percent_ture = count_ture / num_
    print(percent_ture)
    return percent_ture


# -*- coding: utf-8 -*-
# !/usr/bin/env python
# @Time    : 2018/11/17 14:52
# @Author  : xhh
# @Desc    : 余弦相似度计算
# @File    : difference_image_consin.py
# @Software: PyCharm
from PIL import Image
from numpy import average, dot, linalg


# 对图片进行统一化处理
def get_thum(image, size=(64, 64), greyscale=False):
    # 利用image对图像大小重新设置, Image.ANTIALIAS为高质量的
    image = image.resize(size, Image.Resampling.BICUBIC)
    if greyscale:
        # 将图片转换为L模式，其为灰度图，其每个像素用8个bit表示
        image = image.convert('L')
    return image


# 计算图片的余弦距离
def image_similarity_vectors_via_numpy(image1, image2):
    image1 = get_thum(image1)
    image2 = get_thum(image2)
    images = [image1, image2]
    vectors = []
    norms = []
    for image in images:
        vector = []
        for pixel_tuple in image.getdata():
            vector.append(average(pixel_tuple))
        vectors.append(vector)
        # linalg=linear（线性）+algebra（代数），norm则表示范数
        # 求图片的范数？？
        norms.append(linalg.norm(vector, 2))
    a, b = vectors
    a_norm, b_norm = norms
    # dot返回的是点积，对二维数组（矩阵）进行计算
    res = dot(a / a_norm, b / b_norm)
    return res


if __name__ == '__main__':
    # # 两部分对应清理重读  # 应该根据轮廓的相似度
    image_debut = r'F:\pix2pix\dataset_building\zjkj\tst_debut'
    # image_debut_2 = r'F:\data_zjkj\dataset_png_zjkj_clear'

    # # # 人工检测户型
    # for pth_path in os.listdir(image_debut):
    #     print('-----------------------------------------------')
    #     print(pth_path)
    #
    #     path_png = os.path.join(image_debut, pth_path)
    #
    #     array_img = np.array(Image.open(path_png))
    #     show_array(array_img[:, :, 1], str(pth_path))
    #     plt.show()

    # list_outlines_paths = [os.path.join(image_debut_2, pth_path) for pth_path in os.listdir(image_debut_2)]
    list_outlines_paths = []

    for pth_path in tqdm(os.listdir(image_debut)):
        print('-----------------------------------------------')
        print(pth_path)
        path_png = os.path.join(image_debut, pth_path)

        find_similar = False
        similar_path = None

        for pth_path_2 in list_outlines_paths[-10:]:  # 已经存入列表了

            image1 = Image.open(path_png)
            image2 = Image.open(pth_path_2)

            cosin = image_similarity_vectors_via_numpy(image1, image2)

            if cosin >= 0.99:
                find_similar = True
                similar_path = pth_path_2
                break

        if not find_similar:
            print('1')
            list_outlines_paths.append(path_png)
        else:
            if os.path.exists(path_png):
                print('图片余弦相似度', cosin)
                os.remove(path_png)
                # show_array(np.array(image1)[:, :, 1], str(pth_path))
                # plt.show()

            # array_img = np.array(Image.open(similar_path))
            # show_array(array_img[:, :, 1], str(similar_path))
            # plt.show()
            #
            # array_img = np.array(Image.open(path_png))
            # show_array(array_img[:, :, 1], str(pth_path))
            # plt.show()

        # array_img = np.array(Image.open(path_png))
        # show_array(array_img[:, :, 0], str(pth_path))
        #
        # k = np.where(array_img[:,:,0] == 127 )
        # m = list(zip(list(k[0]), list(k[1])))
        # plt.show()
