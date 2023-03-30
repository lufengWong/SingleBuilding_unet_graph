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

import numpy as np


def outer_contours(img):
    contours, hier = cv2.findContours(img, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    return [contours[i] for i in range(len(hier[0])) if hier[0][i][3] == -1]


if __name__ == '__main__':
    print('12')

    path_image_output = r'F:\U-net-train-val-test\test_pkl_had\synth_input_Pkl_output'

    for png in os.listdir(path_image_output):
        name_png = png.split('.')[1]

        path_png = os.path.join(path_image_output, png)

        array_boundary = np.array(Image.open(path_png))[0:256, 0:256, 0]
        array_wall = np.array(Image.open(path_png))[0:256, 256:512, 0]

        # # 显示边界
        # plt.imshow(array_boundary, cmap="gray")
        # plt.title('Boundary-Input')
        # plt.show()

        # 显示生成的线条
        plt.imshow(array_wall, )
        plt.title('Wall-Generate')
        plt.show()

        # kernel_dilate = np.ones((3, 3), np.uint8)
        # array_wall = cv2.dilate(array_wall, kernel_dilate, iterations=1)
        # plt.imshow(array_wall)
        # plt.title('Wall-dilate')
        # plt.show()
        #
        #

        array_wall = np.array(array_wall).reshape(256, 256)
        h, w = array_wall.shape
        rct, array_wall = cv2.threshold(array_wall, 100, 1, cv2.THRESH_BINARY)
        plt.imshow(array_wall)
        plt.title('Wall-value-2')
        plt.show()

        # kernel_erode = np.ones((2, 2), np.uint8)
        # array_wall = cv2.erode(array_wall, kernel_erode, iterations=1)
        # plt.imshow(array_wall)
        # plt.title('Wall-erode')
        # plt.show()

        kernel_dilate = np.ones((3, 3), np.uint8)
        array_wall = cv2.dilate(array_wall, kernel_dilate, iterations=2)
        plt.imshow(array_wall)
        plt.title('Wall-dilate')
        plt.show()

        kernel_erode = np.ones((3, 3), np.uint8)
        array_wall = cv2.erode(array_wall, kernel_erode, iterations=2)
        plt.imshow(array_wall)
        plt.title('Wall-erode')
        plt.show()

        skeleton = morphology.skeletonize(array_wall)
        plt.imshow(skeleton)
        plt.title('Wall-skeleton')
        # plt.show()

        array_wall = np.zeros((256, 256))
        array_wall[skeleton == True] = 1
        array_wall = array_wall.astype(np.uint8)

        contours, hier = cv2.findContours(array_wall, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

        for i in range(len(contours)):
            print('-------------------')
            print(i)
            array_line = np.array(contours[i]).reshape(-1,2)
            # outline = cv2.fitLine(array_line, cv2.DIST_L2, 0, 0.01, 0.01)
            # print(outline)
            plt.plot(array_line[:,0], array_line[:,1])
            plt.show()




        # plt.imshow(array_wall)
        # plt.title('Wall-canny')
        # plt.show()

        # points = np.where(skeleton==True)
        # array_points = np.array( list(zip(points[0], points[1])) )
        # print(array_points)
        # # print(list(zip(points[0], points[1])))
        #

        # # 1 代表应该检测到的行的最小长度
        # lines = cv2.HoughLinesP(array_wall, 2, np.pi / 180, 2)
        # print(lines)
        #
        # for line_index in range(lines.shape[0]):
        #     line = lines[line_index]
        #     array_line = np.array(line).reshape(-1, 2)
        #
        #     plt.plot(array_line[:, 0], array_line[:, 1])
        # plt.show()

        # for x1, y1, x2, y2 in lines[0]:
        #     cv2.line(array_boundary, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # for i in range(len(lines)):
        #     for rho, theta in lines[i]:
        #         a = np.cos(theta)
        #         b = np.sin(theta)
        #         x0 = a * rho
        #         y0 = b * rho
        #         x1 = int(x0 + w * (-b))
        #         y1 = int(y0 + w * (a))
        #         x2 = int(x0 - w * (-b))
        #         y2 = int(y0 - w * (a))
        #
        #         cv2.line(array_wall, (x1, y1), (x2, y2), (0, 0, 255), 2)
    #     # plt.imshow(edges)
    #
    #     plt.imshow(array_wall)
    #     plt.title('line')
    #     # cv2.waitKey()
    #     # cv2.destroyAllWindows()
    #     plt.show()
    #
    # # print('debut')
