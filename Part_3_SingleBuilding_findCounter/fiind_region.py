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


def from_png_2_polygon(path_png):
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
    plt.title('Wall-skeleton')
    plt.show()

    array_wall = np.zeros((utils.size_pix, utils.size_pix))
    array_wall[skeleton == True] = 1
    array_wall = array_wall.astype(np.uint8)

    contours, hierarchy = cv2.findContours(array_wall, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    print(hierarchy)
    print(hierarchy.shape)

    list_largest_boundary = []
    list_rect = []
    list_lines = []

    index_larger = None
    for i in range(hierarchy.shape[1]):
        grade = hierarchy[0, i, :]
        if grade[0] == -1 and grade[3] == -1:
            index_larger = i
            break
    assert index_larger, 'LU--Not find the largest one!'

    for i in range(hierarchy.shape[1]):
        grade = hierarchy[0, i, :]
        points = np.array(contours[i]).reshape(-1, 2)

        if i == index_larger:
            list_largest_boundary.append(points)
            print(points)
        elif grade[3] == index_larger:
            list_rect.append(points)
        else:
            list_lines.append(points)

    return list_largest_boundary, list_rect, list_lines


def write_gemo_txt(list_largest_boundary, list_rect, list_lines, path_txt):
    with open(path_txt, 'w') as f:
        for boundary in list_largest_boundary:
            print('------------Boundary------------------')
            f.write('Boundary')
            f.write('\n')
            print(boundary)
            array_points = np.array(boundary).reshape(-1, 2)
            for index_point in range(array_points.shape[0]):
                point = array_points[index_point, :]

                f.write(str(point[0] * utils.size_grid))
                f.write('\n')
                f.write(str(point[1] * utils.size_grid))
                f.write('\n')

        for rect in list_rect:
            print('------------rect------------------')
            f.write('Rect')
            f.write('\n')
            print(rect)
            array_points = np.array(rect).reshape(-1, 2)
            for index_point in range(array_points.shape[0]):
                point = array_points[index_point, :]

                f.write(str(point[0] * utils.size_grid))
                f.write('\n')
                f.write(str(point[1] * utils.size_grid))
                f.write('\n')

        for line in list_lines:
            print('------------line------------------')
            f.write('Line')
            f.write('\n')
            print(line)
            array_points = np.array(line).reshape(-1, 2)
            for index_point in range(array_points.shape[0]):
                point = array_points[index_point, :]
                f.write(str(point[0] * utils.size_grid))
                f.write('\n')
                f.write(str(point[1] * utils.size_grid))
                f.write('\n')

        f.write('over')
        f.write('\n')

if __name__ == '__main__':
    print('12')

    path_image_output = r'F:\U-net-train-val-test\test_pkl_had\synth_input_Pkl_output'

    for png in os.listdir(path_image_output)[0:1]:
        path_img = os.path.join(path_image_output, png)
        list_largest_boundary, list_rect, list_lines = from_png_2_polygon(path_img)
        path_txt_all = r'C:\Users\Administrator\Desktop\singleBuilding_unet_graph\Part_3_SingleBuilding_findCounter\txt_region_points'
        txt_name = png.split('.')[0]
        write_gemo_txt(list_largest_boundary, list_rect, list_lines, os.path.join(path_txt_all, txt_name + '.txt'))
