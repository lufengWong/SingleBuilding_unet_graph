# -*- coding: utf-8 -*-
# @Time    : 2023/3/8 23:15
# @Author  : Lufeng Wang
# @WeChat  : tofind404
# @File    : stage_1_step_2_twoStep_match.py
# @Software: PyCharm

import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import numpy as np
from PIL import Image

from stage_1_step_1_boundary_via_input import from_input_get_pix
import utils
from function_choose_from_library import get_apartment_elevator_similar, get_region_similar


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


def get_apartment_area(data):
    """
    获取户型面积的列表，获取电梯个数列表
    :param data:
    :return:
    """
    number_apartments = list(set(data[data < utils.value_step]))  # 1,2,3

    number_elevator = list(set(data[data < utils.value_step * 2]) - set(number_apartments))

    apartments_area = [len(np.where(data == id_rec)[0]) * (utils.size_grid ** 2) / (1000 ** 2) for id_rec in
                       number_apartments]

    elevator_count = [len(np.where(data == id_rec)[0]) * (utils.size_grid ** 2) / (1000 ** 2) // (
            (utils.elevator_min_length / 1000) ** 2)
                      if len(np.where(data == id_rec)[0]) * (utils.size_grid ** 2) / (1000 ** 2) // (
            (utils.elevator_min_length / 1000) ** 2) >= 1 else 1
                      for id_rec in number_elevator]  # 保证不会为0

    return [elevator_count, apartments_area]


def compute_centroid(mask, value):
    """

    :param mask: 整个图片
    :param value: 需要计算的mask值
    :return: 中心
    """

    shape_array = mask.shape
    list_x_y = [[h, w] for h in range(shape_array[0]) for w in range(shape_array[1]) if mask[h, w] == value]
    [x_center, y_center] = np.round(np.mean(np.array(list_x_y).reshape(-1, 2), axis=0), decimals=0).tolist()

    return [int(x_center), int(y_center)]


def get_graph_location(points_ploygon_input, apartments_need, num_elevators, paths_png, paths_graph):
    """
    通过两阶段的设计获得匹配的图结构
    :param points_ploygon_input: 输入的多边形角点
    :param apartments_need: 需要户型的面积以及个数
    :param num_elevators: 电梯的数目
    :param paths_png: 数据集图片的地址，列表
    :param paths_graph: 数据集图数据的地址，列表
    :return:
    """
    list_apartments_need = sum([[key] * value for key, value in apartments_need.items()], [])

    # 0 channel boundary 1 channel region
    array_img_input = from_input_get_pix(points_ploygon_input, apartments_need)

    # 获取数据库中纯净的信息 ############
    pngs_all_debut = [name.split('.')[0] for path in paths_png for name in os.listdir(path)]
    graphs_all_debut = [name.split('.')[0] for path in paths_graph for name in os.listdir(path)]
    data_useful = list(set(pngs_all_debut) & set(graphs_all_debut))

    # 获取id-graph数据
    paths_graph_useful = [os.path.join(path, name) for path in paths_graph for name in os.listdir(path)
                          if name.split('.')[0] in data_useful]
    dict_id_graph = {os.path.basename(file).split('.')[0]: np.load(file, allow_pickle=True) for file in
                     paths_graph_useful}

    # 只需图像信息进行两步的匹配
    # step 1 户型的各个面积 电梯的个数 个数需要相同 根据欧式距离进行匹配
    # 将两个通道的数值分阶段相乘 提取第4通道

    # 将两个通道的数值分阶段相乘
    paths_png_useful = [os.path.join(path, name) for path in paths_png for name in os.listdir(path)
                        if name.split('.')[0] in data_useful]

    dict_id_png = {os.path.basename(file).split('.')[0]: np.asarray(Image.open(file)) for file in paths_png_useful}

    list_data_png_useful = [[os.path.basename(img).split('.')[0], np.asarray(Image.open(img))]
                            for img in paths_png_useful]
    dict_id_data = {id_shape: data.astype(np.uint16) for id_shape, data in list_data_png_useful}  # 改变数组的最大值的限制

    dict_data_100channel1_plus_channel2 = {
        key_id: (value[:, :, 1].reshape((utils.size_pix, utils.size_pix)) * utils.value_step
                 + value[:, :, 2].reshape((utils.size_pix, utils.size_pix)) * 1)
        for key_id, value in dict_id_data.items()}

    data_apartment_elevator_to_math = {key_id: get_apartment_area(value) for key_id, value in
                                       dict_data_100channel1_plus_channel2.items()}  # 这一步使用的是面积

    data_region_to_layout = {key_id: np.where(value[:, :, 3] == 255, 1, value[:, :, 3])
                             for key_id, value in dict_id_data.items()}  # 区域信息 第4通道

    list_library_name = [key for key, value in data_apartment_elevator_to_math.items()]

    list_library_apart_elevator = [data_apartment_elevator_to_math[key] for key in list_library_name]
    list_library_apart = [building[1] for building in list_library_apart_elevator]
    list_library_elevator = [int(sum(building[0])) for building in list_library_apart_elevator]

    list_library_region = [data_region_to_layout[key] for key in list_library_name]

    # 开始匹配
    # math_1
    index_chosen_elevator_apartment = get_apartment_elevator_similar(num_elevator=num_elevators,
                                                                     list_target=list_apartments_need,
                                                                     list_num_elevators=list_library_elevator,
                                                                     list_candidates_area=list_library_apart)

    # match_2
    list_ids_candidate = [list_library_name[index] for index in index_chosen_elevator_apartment]
    list_regions_candidate = [list_library_region[index] for index in index_chosen_elevator_apartment]

    regions_chosen = get_region_similar(region_target=array_img_input[:, :, 1],
                                        list_regions_candidations=list_regions_candidate)

    # 处理得到的数据
    list_ids_now = [list_ids_candidate[index] for index in regions_chosen[0]]

    dict_id_rot = {list_ids_candidate[index]: rot for index, rot in zip(regions_chosen[0], regions_chosen[1])}  # 旋转角度

    list_channel_1_plus_3_rot = [
        np.rot90(np.fliplr(dict_data_100channel1_plus_channel2[list_ids_candidate[index]]), rot[1]) if rot[0] == 1
        else np.rot90(dict_data_100channel1_plus_channel2[list_ids_candidate[index]], rot[1])
        for index, rot in zip(regions_chosen[0], regions_chosen[1])]  # 特征图经过旋转

    # for index, feature_chose in enumerate(list_channel_1_plus_3_rot):
    #     show_array(list_channel_1_plus_3_rot[index], str(index))
    #     plt.show()

    # fine tune
    # 提取 中心坐标 提取图结构
    dict_id_graph_location_rot = {}
    dict_id_channel_1_plus_3_rot = {}

    for index, (id, feature_chose) in enumerate(zip(list_ids_now, list_channel_1_plus_3_rot)):

        dict_id_channel_1_plus_3_rot.update({id: feature_chose}) # 旋转后的特征图

        graph = dict_id_graph[id]
        # 处理一下图数据
        graph_clear = [pair.tolist() for pair in graph if (pair[0] != pair[1] and pair[0] < 400 and pair[1] < 400)]
        graph = []
        for pair in graph_clear:
            if [pair[1], pair[0]] not in graph:
                graph.append(pair)

        list_mask_less_300 = [mask for mask in list(set(list(feature_chose.ravel()))) if mask < 400]
        dict_mask_center = {value: compute_centroid(feature_chose, value) for value in list_mask_less_300}

        dict_id_graph_location_rot.update({id: [graph, dict_mask_center]})

    # 最后选中的户型旋转角度的字典(根据面积的交并比) 最后选中户型对应的图结构以及节点的位置(旋转后的), 旋转后的特征图， 所有户型对应的特征图 所有户型对应的图结构
    return list_ids_now, dict_id_rot, dict_id_graph_location_rot, dict_id_channel_1_plus_3_rot, dict_id_png, dict_id_graph


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

    list_ids_now, dict_id_rot, dict_id_graph_location_rot, dict_id_png, dict_id_graph = \
        get_graph_location(points_ploygon_input_test, apartments_need_test, num_elevators_test, paths_png_test, paths_graph_test)

    print('debut')
