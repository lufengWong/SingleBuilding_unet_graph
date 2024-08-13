# -*- coding: utf-8 -*-
# @Time    : 2023/3/8 20:00
# @Author  : Lufeng Wang
# @WeChat  : tofind404
# @File    : stage_1_step_1_boundary_via_input.py
# @Software: PyCharm


from functools import reduce
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import numpy as np
from shapely.geometry import Polygon
import cv2

import utils


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
    patches = [mpatches.Patch(color=colors[i], label="Pixel value {l}".format(l=int(values[i])))
               for i in range(len(values))]
    # put those patched as legend-handles into the legend
    plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., fontsize=7)  # 24

    # plt.grid(True)
    plt.title(name, fontsize=10)  # 20


def get_piex_area_interior(img, mark_target, mark_other):
    """
    使用mark_target标记轮廓内的点，其他用mark_other
    :param img: 图片数组
    :param mark_target: int 0~255
    :param mark_other: int 0~255
    :return:
    """

    kernel = np.ones((3, 3), np.uint8)
    # img = cv2.morphologyEx(img_debut, cv2.MORPH_CLOSE, kernel)  # 不得行
    img = cv2.dilate(img, kernel, iterations=1)  # 先膨胀

    mode = cv2.RETR_EXTERNAL  # 只检测最外围轮廓，包含在外围轮廓内的内围轮廓被忽略
    method = cv2.CHAIN_APPROX_SIMPLE
    contours, hierarchy = cv2.findContours(img, mode, method)
    img_new = cv2.drawContours(img, contours, -1, mark_target, -1)
    other = img_new != mark_target
    other = other * float(mark_other)
    img_new_ = img_new + other

    img_new_ = cv2.erode(img_new_, kernel, iterations=1)  # 再收缩

    return img_new_.astype(np.uint8)


# Python3 program to find an integer point
# on a line segment with given two ends
# function to find gcd of two numbers
def gcd(a, b):
    """
    ChatGPT NB
    GCD是英文greatest common divisor的缩写，意思是最大公约数。
    也就是说，两个或多个数能够整除的最大的正整数。
    例如，15和10的最大公约数是5，因为它们都能被5整除。15/5 = 3，10/5 = 2。
    :param a:
    :param b:
    :return:
    """
    if b == 0:
        return a
    return gcd(b, a % b)


# function to find an integer point on
# the line segment joining pointU and pointV
def findPoint(pointU, pointV):
    # ChatGPT NB
    # get x and y coordinates of points U and V
    # start end. has order
    x1 = pointU[0]
    y1 = pointU[1]
    x2 = pointV[0]
    y2 = pointV[1]

    # get absolute values of x and y differences
    dx = abs(x2 - x1)
    dy = abs(y2 - y1)

    # calculate gcd of dx and dy
    g = gcd(dx, dy)

    # if both differences are 0 then there is no other
    # valid integer point on the line segment other than (x1,y1)
    if g == 0:
        # print("(", x1, ",", y1, "), ", end="")
        return [[x1, y1]]

    # calculate increments in x and y coordinates for each possible solution
    incX = (x2 - x1) // g
    incY = (y2 - y1) // g

    # starting from first possible solution which is closest to U,
    # print points till last possible solution which is closest to V

    # for i in range(g + 1):
    #     print("(", (x1 + i * incX), ",", (y1 + i * incY), "), ", end="")

    return [[x1 + i * incX, y1 + i * incY] for i in range(g + 1)]


def from_input_get_pix(points_ploygon_input, apartments_need,
                       size_pix=utils.size_pix, size_grid=utils.size_grid, length_unit_input_1m=1000):
    """
    从输入的边界坐标更新到图像上
    :param points_ploygon_input: 输入的坐标点
    :param apartments_need: 输入户型需求
    :param num_elevators: 输入电梯的个数
    :param size_pix:图像的大小
    :param size_grid:每个网格的实际尺寸
    :param length_unit_input_1m: 单位 ；1000为 mm
    :return: 所有的边界坐标点
    """
    # 数据的读取并确保数据的合理
    area_input = Polygon(points_ploygon_input).area / (length_unit_input_1m ** 2)
    area_apartments_need = reduce(lambda x, y: x + y[0] * y[1], apartments_need.items(), 0)
    print('The area of input: %s m^2' % area_input)
    print('The area of apartments: %s m^2' % area_apartments_need)
    print('The ratio between apartments area and input area: %f  ' % (area_apartments_need / area_input))
    assert area_apartments_need < area_input, '保证需要的户型面积小于输入的边界的面积'
    points_ploygon_to_pic = (np.asarray(points_ploygon_input).reshape(-1, 2) // size_grid).tolist()
    assert (all(0 <= _ < size_pix for _ in sum(points_ploygon_to_pic, []))), '保证所有的点在图像像素内'

    # 数据的平移
    centroid_point = np.around(np.array(list(Polygon(points_ploygon_to_pic).centroid.coords)[0]), decimals=0).astype(
        int).tolist()
    centroid_pic = [int(size_pix // 2), int(size_pix // 2)]
    x_displace = int(centroid_pic[0] - centroid_point[0])
    y_displace = int(centroid_pic[1] - centroid_point[1])
    points_new_in_pic = (np.array(points_ploygon_to_pic).astype(np.uint8) + np.array([x_displace, y_displace]).astype(
        np.uint8)).tolist()

    # 生成线上的点
    list_lines = [
        [points_new_in_pic[index], points_new_in_pic[index + 1]] if index < len(points_new_in_pic) - 1
        else [points_new_in_pic[index], points_new_in_pic[0]]
        for index in range(len(points_new_in_pic))]

    points_all_in_pic = sum([findPoint(line[0], line[1]) for line in list_lines], [])

    # # 绘图
    # plt.scatter(np.array(points_all_in_pic)[:, 0], np.array(points_all_in_pic)[:, 1])
    # plt.scatter(centroid_pic[0], centroid_pic[1], c='r')
    # plt.xlim([0, size_pix])
    # plt.ylim([0, size_pix])
    # plt.grid()
    # plt.show()

    img_boundary = np.zeros((utils.size_pix, utils.size_pix)).astype(np.uint8)
    for index in points_all_in_pic:
        img_boundary[index[0], index[1]] = 1

    img_region = get_piex_area_interior(img_boundary, 1, 0)
    # # 绘图
    # show_array(img_region, 'img_region')
    # plt.show()

    # kernel_dilate = np.ones((3, 3), np.uint8)
    # kernel_erode = np.ones((3, 3), np.uint8)
    # img_boundary = cv2.dilate(img_boundary, kernel_dilate, iterations=1)  # 先膨胀
    # img_boundary = cv2.erode(img_boundary, kernel_erode, iterations=1)  # 再收缩
    # img_boundary_new = np.zeros((utils.size_pix, utils.size_pix))
    # 再寻找外边界
    # contours, hierarchy = cv2.findContours(img_boundary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    # for index in contours:
    #     img_boundary_new[index[0], index[1]] = 1

    # 绘图
    # show_array(img_boundary, 'boundary')
    # plt.show()

    data_input = np.zeros((utils.size_pix, utils.size_pix, 2))
    data_input[:, :, 0] = img_boundary
    data_input[:, :, 1] = img_region

    return data_input.astype(np.uint8)


if __name__ == '__main__':
    points_ploygon_input_1 = [[0, 5000], [10000, 5000], [10000, 0], [30000, 0], [30000, 20000], [0, 30000]]
    apartments_need_1 = {100: 2, 80: 2, 65: 4}
    num_elevators_1 = 3

    data_need = from_input_get_pix(points_ploygon_input_1, apartments_need_1)
    print(data_need)
