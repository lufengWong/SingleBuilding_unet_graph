# !/usr/bin/env python
# -*-coding:utf-8 -*-

"""
# File       : function_2.py
# Time       ：2022/10/21 19:16
# Author     ：Lufeng Wang
# WeChat     ：tofind404
# Description：
"""
import copy
import os
import shutil
import sys

import PIL.Image
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import scipy

from shapely.geometry import MultiLineString, mapping, MultiPoint
from shapely.geometry import LineString, Polygon, Point, MultiPolygon, MultiPoint
from shapely.geometry import Point
from shapely.ops import split

import numpy as np
import math
from PIL import Image

import ezdxf

import cv2

from sklearn.cluster import DBSCAN, KMeans

from tqdm import tqdm


def get_filelist(dir, Filelist):
    # 遍历文件夹及其子文件夹中的文件，并存储在一个列表中

    # 输入文件夹路径、空文件列表[]

    # 返回 文件列表Filelist,包含文件名（完整路径）

    newDir = dir

    if os.path.isfile(dir):

        Filelist.append(dir)

        # # 若只是要返回文件文，使用这个

        # Filelist.append(os.path.basename(dir))

    elif os.path.isdir(dir):

        for s in os.listdir(dir):
            # 如果需要忽略某些文件夹，使用以下代码

            # if s == "xxx":

            # continue

            newDir = os.path.join(dir, s)

            get_filelist(newDir, Filelist)

    return Filelist


# 注释 apartment = single_building = building_single


def from_dxf_get_polygon(dxf_file_name, layer_name):
    """
    获得图层对应的区域
    Args:
        dxf_file_name: dxf文件名
        layer_name: 图层名（该图岑所在的区域为多边形）
    Returns:
        图层名所在的区域
    """
    doc = ezdxf.readfile(dxf_file_name)
    msp = doc.modelspace()

    # 输入有些为out有些为OUT
    list_layer_names = []
    for layer in doc.layers:
        list_layer_names.append(layer.dxf.name)

    if layer_name == 'out' and layer_name in list_layer_names:
        layer_name = 'out'
    else:
        layer_name = 'OUT'
        if layer_name not in list_layer_names:  # 两种都不存在
            return ((0, 0), (100, 0), (10000, 100), (0, 100)), ((-2000, 0), (-20000, 0), (-20000, 1000), (-2000, 1000))

    # 全部存完再读
    layer_name_1 = layer_name
    group = msp.groupby(dxfattrib='layer')
    layer_cad_1 = group[layer_name_1]

    polyline_all = []
    for line_text in layer_cad_1:
        # print(line_text.dxftype())

        polyline_this = []
        if line_text.dxftype() == 'POLYLINE':

            for point in list(line_text.points()):
                polyline_this.append(tuple(np.around(np.array(list(point)[0:2]), decimals=-1).tolist()))

            if len(polyline_this) >= 4:
                polyline_all.append(tuple(polyline_this))

        elif line_text.dxftype() == 'LWPOLYLINE':
            for point in list(line_text.vertices_in_wcs()):
                polyline_this.append(tuple(np.around(np.array(list(point)[0:2]), decimals=-1).tolist()))

            if len(polyline_this) >= 4:
                polyline_all.append(tuple(polyline_this))

        else:
            print('其他类型的多线段')

            # print(line_text.vertices[:2])
    #         # for point in list(line_text.vertices):
    #         #     print(point.dxf.__doc__)
    #
    # polyline_all = [tuple(tuple(np.around(np.array(list(point)[0:2]), decimals=-1).tolist())
    #                       for point in list(line_text.vertices_in_wcs()))
    #                 for line_text in layer_cad_1]

    return polyline_all


def get_dict_centroid_polygon(list_polygons):
    """
    获得几何点和中心点的字典，以及中心点和几何点的字典
    :param list_polygons:
    :return:
    """

    centroid_all = tuple(
        [tuple(np.around(np.array(list(Polygon(shape).centroid.coords)[0]), decimals=-1).tolist())
         for shape in list_polygons])

    dict_centroid_polygon = dict(zip(centroid_all, list_polygons))
    dict_polygon_centroid = dict(zip(list_polygons, centroid_all))

    return dict_centroid_polygon, dict_polygon_centroid


def get_single_building(dict_centroid_polygon):
    """
    采用密度聚类得到同一张建筑图的建筑单体
    :param dict_centroid_polygon:
    :return:
    """

    # 对质心进行密度聚类
    list_points_centroid = list(dict_centroid_polygon.keys())

    index_apartment_predict = DBSCAN(eps=20000, min_samples=4).fit_predict(list_points_centroid)  # 参数的确定
    # print(index_apartment_predict)
    num_apartment = len(set(index_apartment_predict))
    # print(num_apartment)

    # 得到聚类后的多边形
    list_apartments = []
    for apartment in range(num_apartment):
        apartment_this = []
        for index_apart, centroid_polygon in zip(index_apartment_predict, list_points_centroid):
            if index_apart == apartment:
                apartment_this.append(dict_centroid_polygon[centroid_polygon])
        list_apartments.append(tuple(apartment_this))

    # for index_apart, apartment in enumerate(list_apartments):
    #     for polygon in apartment:
    #         x, y = Polygon(polygon).exterior.xy
    #         plt.plot(x, y)
    #     ax = plt.gca()
    #     ax.set_aspect(1)
    #     plt.title('apartment: %d' % index_apart)
    #     plt.show()

    return tuple(list_apartments)


def get_transfer_building(building_single, dict_polygon_centroid, size_grid, size_pic):
    """
    根据质心将图形平移到图像的放大空间
    :param dict_polygon_centroid:
    :param building_single:  元组
    :param size_grid: 每个网格代表的长度
    :param size_pic: 网格的数目，网格的尺寸
    :return:
    """

    # 将building 元组化
    # 计算原图形的质心坐标

    area_apartments = [Polygon(polygon).area for polygon in building_single]
    centroid_apartment = [dict_polygon_centroid[polygon] for polygon in building_single]

    # print(area_apartments)
    # print(centroid_apartment)

    x_sum = 0
    y_sum = 0
    area_sum = 0
    for area, (x, y) in zip(area_apartments, centroid_apartment):
        x_sum += area * x
        y_sum += area * y
        area_sum += area

    [centroid_building_debut_x, centroid_building_debut_y] = np.around(np.array([x_sum / area_sum, y_sum / area_sum]),
                                                                       decimals=0).tolist()  # 几何中心

    [centroid_ture_scale_x, centroid_ture_scale_y] = [size_grid * (size_pic / 2)] * 2  # 图片(放大后)的几何中心

    dis_x = centroid_ture_scale_x - centroid_building_debut_x  #
    dis_y = centroid_ture_scale_y - centroid_building_debut_y

    # 进行建筑单体的数据进行平移
    list_polygons = []
    for polygon in building_single:
        polygon_this = []
        for point in polygon:
            point_x = point[0] + dis_x
            point_y = point[1] + dis_y
            polygon_this.append((point_x, point_y))
        list_polygons.append(tuple(polygon_this))

    return tuple(list_polygons)


def cluster_area(list_polygons, list_markers_order):
    """
    根据面积对多边形进行聚类, 按照面积从小到大
    :param list_polygons: 多边形列表
    :param list_markers_order: 列表的标记符号 从面积小到大进行排列
    :return:
    """

    dict_categories_space = {}

    # step 1 通过面积的重叠找出公共区以及电梯.楼梯； 根据面积的大小找出来电梯和楼梯
    # 或者根据面积进行聚类
    area_apartments = [(Polygon(polygon).area / 1000) for polygon in list_polygons]
    # 使用面积对房间进行一个排序(从小到大？)
    area_apartments, list_polygons = zip(*sorted(zip(area_apartments, list_polygons)))
    # print(area_apartments)
    # print(list_polygons)

    # 对面积进行聚类
    num_category = len(list_markers_order)
    kmeans = KMeans(n_clusters=num_category, max_iter=1000)
    kmeans.fit(np.array(area_apartments).reshape(-1, 1))
    y_ = kmeans.predict(np.array(area_apartments).reshape(-1, 1))
    # print(y_)

    # 需要得出不同类别的空间的功能，同时给相同类别空间以index mask（Space从1开始，0 for non-room space）

    sort_mark = None
    count_category = -1
    for sort_this, building_ in zip(y_, list_polygons):  # 直接就是一个列表了 #
        # print(sort_this)
        if sort_this != sort_mark:
            # print('新的类别')
            count_category += 1
            category = list_markers_order[count_category]
            dict_categories_space[category] = [building_]  # 没有添加么？0115

            sort_mark = sort_this
        else:
            dict_categories_space[category].append(building_)  # 列表是有顺序的，所以是有index的

    return dict_categories_space


def dis(p1, p2):
    return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5


def isRectangle(list_points_rect):
    """ 判断是否为矩形"""
    p1, p2, p3, p4 = list_points_rect
    x_c = (p1[0] + p2[0] + p3[0] + p4[0]) / 4
    y_c = (p1[1] + p2[1] + p3[1] + p4[1]) / 4
    d1 = dis(p1, (x_c, y_c))
    d2 = dis(p2, (x_c, y_c))
    d3 = dis(p3, (x_c, y_c))
    d4 = dis(p4, (x_c, y_c))
    error_allowed = 100
    # print(d1, d2, d3, d4)
    return abs(d1 - d2) <= error_allowed and abs(d1 - d3) <= error_allowed \
        and abs(d1 - d4) <= error_allowed and abs(d2 - d3) <= error_allowed


# 计算两条边的值
def length2_Rectangle(list_points_rect):
    """ calculate the lengths of two edges"""
    p1, p2, p3, p4 = list_points_rect
    length1 = ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5
    length2 = ((p3[0] - p2[0]) ** 2 + (p3[1] - p2[1]) ** 2) ** 0.5
    return [length1, length2]


def get_single_infor_3(single_building, error_draw):
    """
    获取多个多边形的信息
    :param error_draw:  绘图误差
    :param single_building: 为多边形
    :return:
    """
    order_across_area = [3, 2, 0, 1]  # 电梯 步梯 公共区 套型
    # dict_categories_space = {}

    # 根据重叠找出公共区, 根据公共区找出内部的两类, 对两类进行聚类
    # 20230116 ###########################
    # 根据电梯和楼梯的尺寸模数, 分别为2.0-2，5；2.6-2.8找出电梯和楼梯； 根据图关系邻接所有找出公共区
    # 没有除电梯楼梯外的空白公共区域邻接所有，则为废图纸，舍弃
    # 不是矩形的楼梯， 图纸舍弃
    ######################################

    dict_categories_space = {}

    # step1 - 找出电梯 楼梯
    list_stairs = []
    list_elevators = []
    find_stair = False
    for index_building, polygon_shape_debut in enumerate(single_building):
        polygon_shape = list(Polygon(polygon_shape_debut).exterior.coords)[:-1]
        if len(polygon_shape) == 4:
            # print('4条边')
            if isRectangle(polygon_shape):
                # print('矩形')
                length_min_in_rect = min(length2_Rectangle(polygon_shape))
                # print(length_min_in_rect /1000)

                if 2.6 <= length_min_in_rect / 1000 <= 3.5:
                    list_stairs.append(polygon_shape_debut)
                    find_stair = True
                    print('stair')

                elif 1.9 <= length_min_in_rect / 1000 < 2.6:
                    list_elevators.append(polygon_shape_debut)
                    print('elevator')

    # print(list_elevators)
    # print(list_stairs)

    if (not list_elevators) or (not list_stairs):
        print('Abandon 1 !')
        return None, None
    dict_categories_space[1] = list_elevators
    dict_categories_space[2] = list_stairs

    # 先去掉电梯，楼梯，根据图结构，和所有邻接的是空白公共区
    list_other_polygons = []
    list_polygon_shape_public_zero = []
    for polygon_shape_debut in single_building:
        if (polygon_shape_debut not in list_elevators) and (polygon_shape_debut not in list_stairs):
            list_other_polygons.append(polygon_shape_debut)

    for polygon_shape_may_public in list_other_polygons:
        count_intersect = 0
        shape_polygon_male = Polygon(polygon_shape_may_public).buffer(error_draw, cap_style=3, join_style=2)

        # if polygon_shape_public_zero:
        #     break

        for polygon_shape_debut in single_building:
            shape_polygon_female = Polygon(polygon_shape_debut).buffer(error_draw, cap_style=3, join_style=2)

            # if polygon_shape_public_zero:
            #     break

            if shape_polygon_male.intersects(shape_polygon_female):
                count_intersect += 1

                if count_intersect == len(single_building):
                    list_polygon_shape_public_zero.append(polygon_shape_may_public)

    # print(polygon_shape_public_zero)
    if not list_polygon_shape_public_zero:
        print('Abandon 2 !')
        return None, None
    print('public zero')

    # 取面积较小的为公共空白区
    area_polygon_maybe = [Polygon(shape_may).area for shape_may in list_polygon_shape_public_zero]
    index_min_area = area_polygon_maybe.index(min(area_polygon_maybe))
    dict_categories_space[3] = [list_polygon_shape_public_zero[index_min_area]]

    # list_ladders_lifts = []
    # for index_building_female, polygon_female in enumerate(single_building):
    #     for index_building_male, polygon_male in enumerate(single_building):
    #         if index_building_female == index_building_male:
    #             continue
    #         area_intersections = (Polygon(polygon_female).intersection(Polygon(polygon_male))).area
    #         percent_intersections = area_intersections / Polygon(polygon_female).area  # 小面积（楼梯）占大面积
    #
    #         if area_intersections == 0:
    #             continue
    #
    #         if 1 >= percent_intersections > 0.9:
    #             list_ladders_lifts.append(polygon_female)
    # step2- 对电梯，楼梯进行聚类

    # dict_categories_space = cluster_area(list_ladders_lifts, [2, 1])  # 楼梯 电梯
    #
    # # step2 - 公共区
    # find_public = False
    # for index_building_female, polygon_female in enumerate(single_building):  # 遍历所有的空间
    #     if not find_public:
    #
    #         for key, list_polygons in copy.deepcopy(dict_categories_space).items():  # 电梯和楼梯的内容
    #             if not find_public:
    #
    #                 for polygon_space in list_polygons:
    #                     if not find_public:
    #
    #                         if Polygon(polygon_female) == Polygon(polygon_space):  # 相同的几何图形
    #                             continue
    #                         area_intersections = (Polygon(polygon_female).intersection(Polygon(polygon_space))).area
    #
    #                         if area_intersections == 0:
    #                             continue
    #
    #                         percent_intersections = Polygon(
    #                             polygon_space).area / area_intersections  # 小面积（楼梯）占大面积 100/ 0.1
    #                         if 1 >= percent_intersections > 0.9:
    #                             dict_categories_space[3] = [polygon_female]  # 存入字典
    #                             find_public = True
    #                             break

    #  step3 - 套型
    list_already_polygon = []
    for key, value in dict_categories_space.items():
        for polygon in value:
            list_already_polygon.append(polygon)

    list_polygon_apartment = []
    for polygon in single_building:
        if polygon not in list_already_polygon:
            list_polygon_apartment.append(polygon)

    dict_categories_space[0] = list_polygon_apartment  # 套型

    # 得到图结构(采用膨胀操作) 使用列表进行储存 ###############################
    list_space_graph = []
    for key_female, value_female in dict_categories_space.items():
        for index_polygon_this, polygon_this in enumerate(value_female):
            index_polygon_this += 1  # 从1开始计数
            shape_polygon_this = Polygon(polygon_this).buffer(error_draw, cap_style=3, join_style=2)

            for key_male, value_male in dict_categories_space.items():
                for index_polygon_that, polygon_that in enumerate(value_male):
                    index_polygon_that += 1  # 从1开始计数
                    shape_polygon_that = Polygon(polygon_that).buffer(error_draw, cap_style=3, join_style=2)

                    if shape_polygon_that.intersects(shape_polygon_this):
                        list_space_graph.append(
                            [key_female * 100 + index_polygon_this,
                             key_male * 100 + index_polygon_that])  # 十位为类，各位为index  # 100

    # 需要得到外轮廓,以及填充物的分层

    return dict_categories_space, list_space_graph

    # def get_single_infor(single_building, error_draw):
    #     """
    #     根据面积采用kmeans聚类为四类
    #     :param error_draw:  绘图误差
    #     :param single_building:
    #     :return:
    #     """
    #
    #     # step 1 通过面积的重叠找出公共区以及电梯.楼梯； 根据面积的大小找出来电梯和楼梯
    #     # 或者根据面积进行聚类
    #     area_apartments = [(Polygon(polygon).area / 1000)  for polygon in single_building]
    #     # 使用面积对房间进行一个排序(从小到大？)
    #     area_apartments, single_building = zip(*sorted(zip(area_apartments, single_building)))
    #     print(area_apartments)
    #     print(single_building)
    #
    #     # 对面积进行聚类
    #     num_category = 4
    #     kmeans = KMeans(n_clusters=num_category, max_iter=1000)
    #     kmeans.fit(np.array(area_apartments).reshape(-1, 1))
    #     y_ = kmeans.predict(np.array(area_apartments).reshape(-1, 1))
    #     print(y_)
    #
    #     # 需要得出不同类别的空间的功能，同时给相同类别空间以index mask（Space从1开始，0 for non-room space）
    #
    #     order_across_area = [3, 2, 0, 1]  # 电梯 步梯 公共区 套型
    #     dict_categories_space = {}
    #
    #     sort_mark = None
    #     count_category = -1
    #     for sort_this, building_ in zip(y_, single_building):
    #         print(sort_this)
    #         if sort_this != sort_mark:
    #             print('新的类别')
    #             count_category += 1
    #             category = order_across_area[count_category]
    #             dict_categories_space[category] = [building_]
    #
    #             sort_mark = sort_this
    #         else:
    #             dict_categories_space[category].append(building_)  # 列表是有顺序的，所以是有index的
    #
    #     # 需要得到图结构(采用膨胀操作) 使用列表进行储存
    #     list_space_graph = []
    #     for key_female, value_female in dict_categories_space.items():
    #         for index_polygon_this, polygon_this in enumerate(value_female):
    #             index_polygon_this += 1  # 从1开始计数
    #             shape_polygon_this = Polygon(polygon_this).buffer(error_draw, cap_style=3, join_style=2)
    #
    #             for key_male, value_male in dict_categories_space.items():
    #                 for index_polygon_that, polygon_that in enumerate(value_male):
    #                     index_polygon_that += 1  # 从1开始计数
    #                     shape_polygon_that = Polygon(polygon_that).buffer(error_draw, cap_style=3, join_style=2)
    #
    #                     if shape_polygon_that.intersects(shape_polygon_this):
    #                         list_space_graph.append(
    #                             [key_female * 10 + index_polygon_this, key_male * 10 + index_polygon_that])  # 十位为类，各位为index
    #
    #     # 需要得到外轮廓,以及填充物的分层
    #
    #     return dict_categories_space, list_space_graph


def get_piex_line(size_img, size_grid, lines_list):
    """
    将对应线段的的值变为255,并返回矩阵
    :param size_img:
    :param size_grid:
    :param lines_list: shapely 样式
    :return:
    """
    img = np.zeros((size_img, size_img))
    # print(img)

    # 除网格的大小 ######在这里变为网格的尺寸了 # 这里取整数的问题
    array_lines_ceil = np.ceil(np.array(lines_list).reshape(-1, 4) / size_grid).astype(int)  # 数字变小
    # print(array_lines_ceil)
    array_lines_floor = np.floor(np.array(lines_list).reshape(-1, 4) / size_grid).astype(int)  # 数字变小
    # print(array_lines_floor)
    array_lines = np.vstack((array_lines_ceil, array_lines_floor))

    # array_lines = array_lines_floor

    def line_2_piexIndex(line):
        """
        将起止点的线变为离散的点
        :param line:
        :return:
        """
        [x_debut, y_debut, x_end, y_end] = line

        num_x = int(abs(x_debut - x_end + 1))
        num_y = int(abs(y_debut - y_end + 1))

        max_num = max(num_x, num_y)
        x_line_space = np.linspace(x_debut, x_end, max_num, endpoint=True, dtype=int)
        y_line_space = np.linspace(y_debut, y_end, max_num, endpoint=True, dtype=int)

        points_line = list(zip(x_line_space, y_line_space))
        return points_line

    index_piex = [line_2_piexIndex(line) for line in array_lines[:, 0:4]]
    index_all_piex = sum(index_piex, [])  # 所有线条的合并为点

    # print(index_all_piex)

    def assign(index):
        # print(index)
        img[index[0], index[1]] = 255
        return img

    # 迭代器
    list(map(assign, index_all_piex))  # 必须list了

    # 形态学操作-闭操作
    kernel = np.ones((3, 3), np.uint8)
    closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)

    return closing.astype(np.uint8)


def get_piex_area_interior(img, mark_target, mark_other):
    """
    使用mark_target标记轮廓内的点，其他用mark_other
    :param img: 图片数组
    :param mark_target: int 0~255
    :param mark_other: int 0~255
    :return:
    """
    mode = cv2.RETR_EXTERNAL  # 只检测最外围轮廓，包含在外围轮廓内的内围轮廓被忽略
    method = cv2.CHAIN_APPROX_SIMPLE
    contours, hierarchy = cv2.findContours(img, mode, method)
    img_new = cv2.drawContours(img, contours, -1, mark_target, -1)
    other = img_new != mark_target
    other = other * float(mark_other)
    img_new_ = img_new + other
    return img_new_.astype(np.uint8)


def get_piex_outline(img, mark_target_exterior, mark_interior_wall, mark_other):
    """
    得到最外围轮廓，并使用mark_target_exterior标记
    输入为矩阵，线用非零元素标记，其他用零标记
    输出为矩阵，轮廓内的线用mark_interior_wall标记，其他用mark_other标记
    :param img: 图片数组
    :param mark_target_exterior: int 0~255
    :param mark_interior_wall: int 0~255
    :param mark_other: int 0~255
    :return:
    """
    mode = cv2.RETR_EXTERNAL  # 只检测最外围轮廓，包含在外围轮廓内的内围轮廓被忽略
    method = cv2.CHAIN_APPROX_NONE
    contours, hierarchy = cv2.findContours(img, mode, method)
    # print(contours)  # 此时其交换

    if len(contours) == 1:
        contours = np.squeeze(contours)  # 从数组的形状中删除单维度条目，即把shape中为1的维度去掉
        contours = contours[:, [1, 0]]  # 交换x和y坐标，因为opencv的原点
        # plt.plot(np.array(contours)[:, 0], np.array(contours)[:, 1], "r")
        # plt.show()
        img_out = np.ones(np.shape(img)) * mark_other  # 标记其他

        def assign(location):
            (x, y) = location
            img_out[x, y] = mark_target_exterior
            return img_out

        list(map(assign, contours))

        img_out_copy = np.ones(np.shape(img)) * mark_other
        img_out_copy[img_out == mark_target_exterior] = 255  # 原始为255

        # # 相减
        # 此处做个膨胀操作， 删除边界
        kernel = np.ones((5, 5), np.uint8)
        dilation = cv2.dilate(img_out_copy, kernel, iterations=1)

        img_in = img - dilation
        img_in[img_in == -255] = 0
        img_in[img_in == 255] = mark_interior_wall

        return img_out.astype(np.uint8), img_in.astype(np.uint8)
    else:
        print("存在多条外围轮廓，请检查！")
        sys.exit()


def get_lines(polygon_coord):
    polygon = Polygon(polygon_coord)
    list_coord = list(polygon.exterior.coords)
    lines = [[list_coord[index], list_coord[index + 1]] for index in range(len(list_coord) - 1)]
    return lines


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
    patches = [mpatches.Patch(color=colors[i], label="Pixel value {l}".format(l=int(values[i]))) for i in
               range(len(values))]
    # put those patched as legend-handles into the legend
    plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., fontsize=7)  # 24

    # plt.grid(True)
    plt.title(name, fontsize=10)  # 20


def get_piex_need(dict_categories_space, pic_size, grid_size):
    """
    根据KJL将将图片设计为四个通道
    :param grid_size:  网格的大小
    :param pic_size: 图片的大小
    :param dict_categories_space: 储存区域的字典
    :return:
    """

    img_4_channel = np.zeros((pic_size, pic_size, 4))  # 进行数据的更换

    # 将所有的多边形处理为线条 并去重(不去重其实也无所谓)
    list_polygons_all = [value for key, list_value in dict_categories_space.items() for value in list_value]
    list_lines_all = [get_lines(polygon_) for polygon_ in list_polygons_all]  # 需要继续展开
    list_lines = sum(list_lines_all, [])

    array_img_all_lines = get_piex_line(pic_size, grid_size, list_lines)

    # plt.imshow(array_img_all_lines, cmap='rainbow')
    # plt.title('debut_all_lines')
    # plt.show()

    # cv2.imshow('debut_all_lines', array_img_all_lines)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # channel-1: exterior wall 127, other 0
    array_img_exterior_wall, array_img_interior_wall = get_piex_outline(array_img_all_lines, 127, 60, 0)  # 背景为0

    # plt.imshow(array_img_exterior_wall, cmap='rainbow')
    # plt.title('out')
    # plt.show()
    #
    # plt.imshow(array_img_interior_wall, cmap='rainbow')
    # plt.title('in')
    # plt.show()

    # cv2.imshow('exterior_wall', array_img_exterior_wall)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # cv2.imshow('in_wall', array_img_interior_wall)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    img_4_channel[:, :, 0] = array_img_exterior_wall
    show_array(img_4_channel[:, :, 0], 'Channel-1')
    # plt.show()
    plt.pause(0.5)
    plt.close()

    # channel-4 Exterior area 0, interior 255
    array_img_exterior_area = get_piex_area_interior(array_img_exterior_wall, 255, 0)
    img_4_channel[:, :, 3] = array_img_exterior_area

    show_array(img_4_channel[:, :, 3], 'Channel-4')
    # plt.show()
    plt.pause(0.5)
    plt.close()

    # channel-2 多种空间
    # channel-3 一起来
    # array_img_channel_2 = np.zeros((pic_size, pic_size))  # 默认为0， 所以内部空挡默认为0，为公寓
    array_img_channel_2 = np.full((pic_size, pic_size), 4)
    array_img_channel_3 = np.zeros((pic_size, pic_size))

    # 处理区域
    def get_array_imgs(index_space):
        """
        返回目标区域 （内部）
        :param index_space:
        :return:
        """
        list_polygons = dict_categories_space[index_space]
        list_array_imgs = []  # 可能有多个
        # print('222222222222222222222222222222222222222222222222222222222222222222222222222')
        # print(index_space)
        # print(len(list_polygons))

        for polygon in list_polygons:
            lines = get_lines(polygon)
            array_img_lines = get_piex_line(pic_size, grid_size, lines)
            array_img_category = get_piex_area_interior(array_img_lines, 255, 0)  # 目标值为255

            list_array_imgs.append(array_img_category)

        return list_array_imgs

    # 开始处理
    list_space_index = [3, 2, 1, 0]  # 公共区会被覆盖
    for index_space_count in list_space_index:
        list_array_imgs_ = get_array_imgs(index_space_count)  # 目标的值为255  # 可能有多个图形
        for index_count, img in enumerate(list_array_imgs_):
            index_count += 1  # 对其进行分类
            array_img_channel_2[img == 255] = int(index_space_count)  # 类别通道

            array_img_channel_3[img == 255] = int(index_count)  # 计算数目的通道

    array_img_channel_2[array_img_exterior_area == 0] = 4
    array_img_channel_2[img_4_channel[:, :, 0] == 127] = 5
    array_img_channel_2[array_img_interior_wall == 60] = 6

    img_4_channel[:, :, 1] = array_img_channel_2.astype(np.uint8)
    img_4_channel[:, :, 2] = array_img_channel_3.astype(np.uint8)

    show_array(img_4_channel[:, :, 1], 'Channel-2')
    # plt.show()
    plt.pause(0.5)
    plt.close()

    show_array(img_4_channel[:, :, 2], 'Channel-3')
    # plt.show()
    plt.pause(0.5)
    plt.close()

    return img_4_channel


def make_piex(dir_dxf, name_dxf, dir_png, dir_graph, new_name):
    # 输入文件参数 以及 图像参数
    name_layer = 'out'

    # dict_spaces_category = [(0, 'PublicSpace'), (1, 'Apartment'), (2, 'Ladder'), (3, 'Lift'),
    #                         (4, 'ExternalArea'), (5, 'ExteriorWall'), (6, 'InteriorWall')]

    size_grid = 192  ########################
    size_pic = 256
    error_plot = 250

    # 从文件读取数据

    polygons_all = from_dxf_get_polygon(os.path.join(dir_dxf, name_dxf), name_layer)

    # 计算质心
    dict_centroid_polygon, dict_polygon_centroid = get_dict_centroid_polygon(polygons_all)

    for centroid, polygon in dict_centroid_polygon.items():
        x, y = Polygon(polygon).exterior.xy
        plt.plot(x, y)
        plt.scatter(centroid[0], centroid[1])

    ax = plt.gca()
    ax.set_aspect(1)
    plt.title('DXF')
    # plt.show()
    plt.pause(0.5)
    plt.close()

    # # 对数据进行聚类 得到一个CAD的多个building
    # list_building_single = get_single_building(dict_centroid_polygon)
    # # print(list_building_single)
    #
    # # 以第一个building为例进行数据的处理，将来可以处理为一个图纸数据并行的模式，包括多个图纸也可采用并行
    # building_this = list_building_single[0]

    # 根据这个此栋楼的质心平移到256图像的实际尺寸（256*256）的质心位置，进行平移
    #          # 得到每个空间的质心的信息

    building_this = polygons_all  # 每张图一个
    dict_centroid_polygon_building, dict_polygon_centroid_building = get_dict_centroid_polygon(building_this)

    # # 绘图显示
    # for centroid, polygon in dict_centroid_polygon_building.items():
    #     x, y = Polygon(polygon).exterior.xy
    #     plt.plot(x, y)
    #     plt.scatter(centroid[0], centroid[1])
    #
    # ax = plt.gca()
    # ax.set_aspect(1)
    # plt.title('single')
    # plt.show()

    #          # 平移主函数
    building_transfer = get_transfer_building(building_this, dict_polygon_centroid_building, size_grid, size_pic)

    # for polygon in building_transfer:
    #     x, y = Polygon(polygon).exterior.xy
    #     plt.plot(np.true_divide(x, size_grid), np.true_divide(y, size_grid))  # 在256 空间显示
    #
    # plt.scatter([0, size_pic], [0, size_pic])
    # ax = plt.gca()
    # ax.set_aspect(1)
    # plt.title('Transfer')
    # plt.show()

    #          # 得到每个平移后空间的质心信息 #save1
    dict_centroid_polygon_building_tran, dict_polygon_centroid_building_tran = get_dict_centroid_polygon(
        building_transfer)

    # for centroid, polygon in dict_centroid_polygon_building_tran.items():
    #     x, y = Polygon(polygon).exterior.xy
    #     plt.plot(x, y)
    #     plt.scatter(centroid[0], centroid[1])
    #
    # ax = plt.gca()
    # ax.set_aspect(1)
    # plt.title('DXF')
    # plt.show()

    # 得到这个所有的单体信息（字典）以及图结构（列表） # save 2
    dict_categories_space_this, list_space_graph_this = get_single_infor_3(building_transfer, error_draw=error_plot)

    if (not dict_categories_space_this) or (not list_space_graph_this):
        return 0

    # 根据单体信息得到4通道的数据 # save3

    img_4_channel = get_piex_need(dict_categories_space_this, pic_size=size_pic, grid_size=size_grid).astype(np.uint8)

    # for reverse_choose in [1, 0]:  #
    for reverse_choose in [1]:  #
        if reverse_choose == 1:
            array_change = img_4_channel
        else:
            array_change = np.flip(img_4_channel, 0)

        # for angle_choose in [1, -1, 2, 3]:
        for angle_choose in [1]:
            array_change_2 = np.rot90(array_change, angle_choose)

            # name_png = (name_dxf.split('.')[0]) + str('_') + str(reverse_choose) + str('_') + str(angle_choose)

            name_ = str(new_name) + str('_') + str(reverse_choose) + str('_') + str(angle_choose)

            name_png = name_ + '.png'

            Image.fromarray(np.uint8(array_change_2)).save(os.path.join(dir_png, str(name_png)))
            # print(list_space_graph_this)

            name_graph = name_ + '.npy'
            np.save(os.path.join(dir_graph, str(name_graph)), np.array(list_space_graph_this, dtype=object))

    return 666


if __name__ == '__main__':

    dxf_file = r'F:\data_zjkj\CAD_layout'

    png_file_zjkj = r'F:\data_zjkj\dataset_png_zjkj'
    graph_file_zjkj = r'F:\data_zjkj\dataset_graph_zjkj'

    png_file_other = r'F:\data_zjkj\dataset_png_other'
    graph_file_other = r'F:\data_zjkj\dataset_graph_other'

    txt_log_failed = r'F:\data_zjkj\log_make'

    count_debut = 1011

    if not count_debut:
        if os.path.exists(png_file_zjkj):
            shutil.rmtree(png_file_zjkj)
        os.mkdir(png_file_zjkj)

        if os.path.exists(png_file_other):
            shutil.rmtree(png_file_other)
        os.mkdir(png_file_other)

        if os.path.exists(graph_file_zjkj):
            shutil.rmtree(graph_file_zjkj)
        os.mkdir(graph_file_zjkj)

        if os.path.exists(graph_file_other):
            shutil.rmtree(graph_file_other)
        os.mkdir(graph_file_other)

        if os.path.exists(txt_log_failed):
            shutil.rmtree(txt_log_failed)
        os.mkdir(txt_log_failed)

        file_log_success = open(os.path.join(txt_log_failed, 'log_success.txt'), 'w')
        file_log_fail = open(os.path.join(txt_log_failed, 'log_fail.txt'), 'w')

    else:
        file_log_success = open(os.path.join(txt_log_failed, 'log_success.txt'), 'a')
        file_log_fail = open(os.path.join(txt_log_failed, 'log_fail.txt'), 'a')

    # 读取所有的文件
    list_all_files = get_filelist(dxf_file, [])
    list_file_address_name = []
    for file_may in list_all_files:
        if os.path.split(file_may)[1].split('.')[-1] == 'dxf':
            list_file_address_name.append(os.path.split(file_may))

    # list_dxf_names = [name for name in os.listdir(dxf_file) if name.split('.')[-1] == 'dxf']
    #######

    count_all = 0
    print(f'Number of dataset: {len(list_file_address_name)}')
    for [dxf_address, name] in tqdm(list_file_address_name):

        count_all += 1
        if count_all < count_debut:
            continue

        if name.split('T')[0] == "1":
            png_file = png_file_zjkj
            graph_file = graph_file_zjkj
        else:
            png_file = png_file_other
            graph_file = graph_file_other

        # print('****************************************')
        print(count_all)
        print(os.path.join(dxf_address, name))

        # 先判断是否有out或者OUT ############
        dxf_file_name = os.path.join(dxf_address, name)
        doc = ezdxf.readfile(dxf_file_name)

        # 输入有些为out有些为OUT
        list_layer_names = [layer.dxf.name for layer in doc.layers]

        # if ('OUT' in list_layer_names) or ('out' in list_layer_names):
        #     result = make_piex(dir_dxf=dxf_address, name_dxf=name, dir_png=png_file, dir_graph=graph_file,
        #                        new_name=count_all)
        # else:
        #     result = None

        result = make_piex(dir_dxf=dxf_address, name_dxf=name, dir_png=png_file, dir_graph=graph_file,
                           new_name=count_all)

        if result:
            file_log_success.write('\n')
            file_log_success.write(str(count_all))
            file_log_success.write('\n')
            file_log_success.write(str(os.path.join(dxf_address, name)))
            file_log_success.flush()
        else:
            file_log_fail.write('\n')
            file_log_fail.write(str(count_all))
            file_log_fail.write('\n')
            file_log_fail.write(str(os.path.join(dxf_address, name)))
            file_log_fail.flush()

    # file_log_success.write('\n')
    # file_log_success.write('\n')
    # file_log_success.write('***** 成功录入个数为：%d * 4 *****' % count_piex_success)
    file_log_success.close()

    # file_log_fail.write('\n')
    # file_log_success.write('\n')
    # file_log_fail.write('***** 失败录入个数为：%d *****' % count_piex_fail)
    file_log_fail.close()
