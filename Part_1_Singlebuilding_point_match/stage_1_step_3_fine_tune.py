# -*- coding: utf-8 -*-
# @Time    : 2023/3/9 22:29
# @Author  : Lufeng Wang
# @WeChat  : tofind404
# @File    : stage_1_step_3_fine_tune.py
# @Software: PyCharm

import os
import collections
import math
from collections import Counter

from functools import reduce
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import numpy as np

from shapely.geometry import MultiLineString, mapping, MultiPoint
from shapely.geometry import LineString, Polygon, Point, MultiPolygon, MultiPoint
from shapely.geometry import Point
from shapely.ops import split

from PIL import Image

import utils
from stage_1_step_2_twoStep_match_2 import get_graph_location
from stage_1_step_1_boundary_via_input import from_input_get_pix
from function_fine_tune_2 import from_input_get_one_finetune


# 排序坐标点 ###########################################

def distance(p1, p2):
    # 使用勾股定理，返回两点间的欧氏距离
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def angle(p1, p2):
    # 使用反正切函数，返回两点间的夹角（弧度制）
    return math.atan2(p1[1] - p2[1], p1[0] - p2[0])


def centroid(points):
    # 对所有点的横坐标求和，然后除以点的个数，得到重心的横坐标
    x = sum(p[0] for p in points) / len(points)  # 对所有点的纵坐标求和，然后除以点的个数，得到重心的纵坐标
    y = sum(p[1] for p in points) / len(points)  # 返回重心的坐标（元组形式） return (x, y)
    return x, y


def clockwise_sort(points):
    # 计算重心
    c = centroid(points)
    # 按照极角排序，使用lambda表达式作为排序依据
    points.sort(key=lambda p: angle(c, p))
    # 返回排序后的点（列表形式）
    return points


# ###########################################

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


def rotate_point(point_debut, point_center, alpha, direction):
    """
    一个点绕一个点进行旋转
    :param point_debut: 初始点的位置
    :param point_center: 旋转中心
    :param alpha: 旋转的角度 （角度制）
    :param direction: -1 为逆时针 1 为顺时针
    :return: 新的点的位置
    """

    [a, b] = point_center
    [x, y] = point_debut
    beta = alpha * np.pi / 180
    # direction为1表示顺时针旋转，为-1表示逆时针旋转
    c = np.round((x - a) * math.cos(beta) + direction * (y - b) * math.sin(beta) + a, decimals=0)
    d = np.round((y - b) * math.cos(beta) - direction * (x - a) * math.sin(beta) + b, decimals=0)
    return [int(c), int(d)]


def flip_point_left_right(point_debut, point_center):
    """
    点绕着某点进行左右翻转
    :param point_debut: 原始点的位置
    :param point_center: 点对称的点
    :return:
    """
    x_flip = np.round(point_center[0] * 2 - point_debut[0], decimals=0)  # x对称
    y_flip = np.round(point_debut[1], decimals=0)  # y相等
    return [x_flip, y_flip]


def get_area_apartments(array_data):
    """

    :param array_data: 特征图
    :return: 每个户型mask对应的面积
    """

    # number_apartments = list(set(array_data[array_data < utils.value_step * 1]))  # 1,2,3
    # list_index_apartments = [100 * 0 + index for index in number_apartments]

    list_index_apartments = list(set(array_data[array_data < utils.value_step * 1]))

    dict_apartment_area = {
        index: np.round(len(np.where(array_data == index)[0]) * (utils.size_grid ** 2) / (1000 ** 2), decimals=0)
        for index in list_index_apartments}

    return dict_apartment_area


def get_area_public(array_data):
    """

    :param array_data: 特征图
    :return: 每个户型mask对应的面积
    """

    # number_apartments = list(set(array_data[array_data < utils.value_step * 1]))  # 1,2,3
    # list_index_apartments = [100 * 0 + index for index in number_apartments]

    list_index_apartments = list(
        set(array_data[(utils.value_step * 1 < array_data) & (array_data < utils.value_step * 4)]))

    dict_apartment_area = {
        index: np.round(len(np.where(array_data == index)[0]) * (utils.size_grid ** 2) / (1000 ** 2), decimals=0)
        for index in list_index_apartments}

    return dict_apartment_area


def merge_dict(x, y):
    # 将两个字典转换为Counter对象，并进行加法运算，得到一个新的Counter对象
    c = Counter(x) + Counter(y)
    # 将Counter对象转换为普通字典，并返回
    return dict(c)


if __name__ == '__main__':
    print('12')

    # 输入的信息 ######################
    points_ploygon_input_test = [[0, 5000], [10000, 5000], [10000, 0], [30000, 0], [30000, 20000], [0, 30000]]
    apartments_need_test = {100: 2, 80: 2, 65: 2}
    num_elevators_test = 3

    paths_png_test = [r'F:\building_piex_20230221\dataset_png_other_clear',
                      r'F:\building_piex_20230221\dataset_png_zjkj_clear']

    paths_graph_test = [r'F:\building_piex_20230221\dataset_graph_zjkj',
                        r'F:\building_piex_20230221\dataset_graph_other']

    # 得到的图像种边界和区域
    img_input = from_input_get_pix(points_ploygon_input_test, apartments_need_test)
    img_boundary = img_input[:, :, 0]
    img_region = img_input[:, :, 1]

    # 绘制区域
    show_array(img_region, 'region')

    list_index_polygon = np.where(img_boundary == 1)
    index_ploygon_in_pix = clockwise_sort([[x, y] for x, y in zip(list_index_polygon[0], list_index_polygon[1])])

    # 绘制像素点组成的边界
    x, y = Polygon(index_ploygon_in_pix).exterior.xy  # polygon
    plt.plot(y, x)  # 为了show_array匹配
    plt.show()

    # 得到匹配的信息
    list_ids_now, dict_id_rot, dict_id_graph_location_rot, dict_id_channel_1_plus_3_rot, dict_id_png, dict_id_graph = \
        get_graph_location(points_ploygon_input_test, apartments_need_test, num_elevators_test, paths_png_test,
                           paths_graph_test)

    # 进行微调
    # 大概分为三步
    # 匹配每个面积对应的点
    # 根据 空白公共区 楼梯 电梯 大面积套型 小面积套型 进行依次的旋转

    for id in list_ids_now[0:3]:  # 遍历所选出来的
        feature_map = dict_id_channel_1_plus_3_rot[id]

        # 绘制特征图
        show_array(feature_map, str(id))
        # plt.show()

        graph_map = dict_id_graph_location_rot[id]  # 是不变的 ##########
        dict_apartment_area = get_area_apartments(feature_map)
        dict_public_area = get_area_public(feature_map)  # 公共区的面积
        sorted_dict_apartment_area = sorted(dict_apartment_area.items(), key=lambda x: x[1])

        # 计算原来的面积并排序
        sorted_list_apartments_area = sorted(sum([[key] * value for key, value in apartments_need_test.items()], []))

        # 新的最终的户型对应的数据
        dict_index_area_ture_apartment = {id_area[0]: area for id_area, area in
                                          zip(sorted_dict_apartment_area,
                                              sorted_list_apartments_area)}  # 匹配后的点对应的户型面积  #####

        graph_edge = graph_map[0]  # 边的邻接关系
        nodes_location = graph_map[1]  # 点的位置
        dict_area = merge_dict(dict_public_area, dict_index_area_ture_apartment)

        # 绘制初始图的位置
        for key, value in nodes_location.items():
            plt.scatter(value[1], value[0])  # 为了和show_array 匹配
            plt.text(value[1], value[0], str(key))  # 为了和show_array 匹配
        plt.show()

        list_nodes_exceed = []  # 超出边界范围的坐标点
        list_nodes_contain = []  # 在边界范围的坐标点

        for key, value in nodes_location.items():
            if Polygon(index_ploygon_in_pix).contains(Point(value)):
                list_nodes_contain.append(key)
            else:
                list_nodes_exceed.append(key)

        # 做一个排序
        list_nodes_exceed = sorted(list_nodes_exceed, reverse=True)

        # 绘制像素点组成的边界 (都是在正常坐标系里边)
        # 绘制区域 （小大坐标系为准了）
        show_array(img_region, 'region')

        point_new = {key: [value[0], value[1]] for key, value in nodes_location.items()}  # 转换x,y

        # 绘制原来的点
        for key, value in point_new.items():
            plt.scatter(value[1], value[0])  # 为了和show_array 匹配
            plt.text(value[1], value[0], str(key))  # 为了和show_array 匹配
        plt.show()

        for point in list_nodes_exceed:
            print(point)
            point_new = from_input_get_one_finetune(points_ploygon=index_ploygon_in_pix,
                                                    nodes_pos_debut=point_new,  # x,y 需要换位置###########################
                                                    nodes_area=dict_area,
                                                    nodes_adjacent=graph_edge,
                                                    node_move=point,
                                                    roi_mask_size=10,
                                                    k_reject=0.1)

            # 显式变换后的点
            for key, value in point_new.items():
                plt.scatter(value[1], value[0])  # 为了和show_array 匹配
                plt.text(value[1], value[0], str(key))  # 为了和show_array 匹配
            plt.show()

            # 循环至完毕
            for key, value in point_new.items():
                if Polygon(index_ploygon_in_pix).contains(Point(value)):
                    list_nodes_contain.append(key)
                else:
                    list_nodes_exceed.append(key)

        # 绘制像素点组成的边界 (都是在正常坐标系里边)
        # 绘制区域 （小大坐标系为准了）
        show_array(img_region, 'region')

        # 绘制原来的点
        for key, value in point_new.items():
            plt.scatter(value[1], value[0])  # 为了和show_array 匹配
            plt.text(value[1], value[0], str(key))  # 为了和show_array 匹配
        plt.show()

    print('debut')
