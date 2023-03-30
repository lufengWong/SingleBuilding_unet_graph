# -*- coding: utf-8 -*-
# @Time    : 2023/3/28 20:07
# @Author  : Lufeng Wang
# @WeChat  : tofind404
# @File    : ex1.py
# @Software: PyCharm
import os

import numpy as np
import scipy

import matplotlib.pyplot as plt
from PIL import Image
from skimage import morphology, draw

import cv2


import utils

def outer_contours(img):
    contours, hier = cv2.findContours(img, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    return [contours[i] for i in range(len(hier[0])) if hier[0][i][3] == -1]


def find_region(path_imgs):
    for png in os.listdir(path_imgs):



        name_png = png.split('.')[0]

        path_png = os.path.join(path_image_output, png)

        array_boundary = np.array(Image.open(path_png))[0:256, 0:256, 0]
        array_wall = np.array(Image.open(path_png))[0:256, 256:512, 0]

        # 显示生成的线条
        plt.imshow(array_wall, )
        plt.title('Wall-Generate')
        plt.show()

        array_wall = np.array(array_wall).reshape(256, 256)
        h, w = array_wall.shape
        rct, array_wall = cv2.threshold(array_wall, 100, 1, cv2.THRESH_BINARY)
        plt.imshow(array_wall)
        # plt.title('Wall-value-2')
        # plt.show()

        kernel_dilate = np.ones((3, 3), np.uint8)
        array_wall = cv2.dilate(array_wall, kernel_dilate, iterations=2)
        plt.imshow(array_wall)
        # plt.title('Wall-dilate')
        # plt.show()

        kernel_erode = np.ones((3, 3), np.uint8)
        array_wall = cv2.erode(array_wall, kernel_erode, iterations=2)
        plt.imshow(array_wall)
        # plt.title('Wall-erode')
        # plt.show()

        skeleton = morphology.skeletonize(array_wall)
        plt.imshow(skeleton)
        # plt.title('Wall-skeleton')
        # plt.show()

        array_wall = np.zeros((utils.size_pix, utils.size_pix))
        array_wall[skeleton == True] = 1
        array_wall = array_wall.astype(np.uint8)

        contours, hier = cv2.findContours(array_wall, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        with open(name_png+'.txt', 'w') as f:
            for i in range(len(contours)):
                f.write('Points')
                f.write('\n')
                array_line = np.array(contours[i]).reshape(-1, 2)
                for line_index in range(array_line.shape[0]):
                    line = array_line[line_index, :]
                    f.write(str(line[0] * utils.size_grid))
                    f.write('\n')
                    f.write(str(line[1]  * utils.size_grid))
                    f.write('\n')
                # outline = cv2.fitLine(array_line, cv2.DIST_L2, 0, 0.01, 0.01)
                # # print(outline)
                plt.plot(array_line[:, 0], array_line[:, 1])

        plt.show()





if __name__ == '__main__':
    print('12')

    path_image_output = r'F:\U-net-train-val-test\test_pkl_had\synth_input_Pkl_output'
    find_region(path_image_output)
