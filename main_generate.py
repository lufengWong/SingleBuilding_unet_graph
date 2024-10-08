# -*- coding: utf-8 -*-
# @Time    : 2023/4/3 14:49
# @Author  : Lufeng Wang
# @WeChat  : tofind404
# @File    : main_generate.py
# @Software: PyCharm


import os
import sys
import shutil

import matplotlib.pyplot as plt

sys.path.append(os.getcwd())
sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), 'Part_1_Singlebuilding_point_unet_align'))
sys.path.append(os.path.join(os.getcwd(), 'Part_2_SingleBuilding_Unet'))
# for home, dirs, files in os.walk(os.path.join(os.getcwd(), 'Demo2_software_V0')):
#     sys.path.append(home)

import numpy as np
import torch
import pickle

import utils
from Part_1_Singlebuilding_point_unet_align import stage_1_step_1_boundary_output
from Part_1_Singlebuilding_point_unet_align import stage_1_step_4_generate_input
from Part_2_SingleBuilding_Unet import generate_synth
from Part_3_SingleBuilding_findCounter import find_region

if __name__ == '__main__':
    print('12')

    # 数据库
    # 平面图库
    paths_png_test_1 = [r'F:\Dataset_zjkj_4_channel_graph\building_piex_20230221\dataset_png_other_clear',
                        r'F:\Dataset_zjkj_4_channel_graph\building_piex_20230221\dataset_png_zjkj_clear']

    # 图结构库
    paths_graph_test_1 = [r'F:\Dataset_zjkj_4_channel_graph\building_piex_20230221\dataset_graph_zjkj',
                          r'F:\Dataset_zjkj_4_channel_graph\building_piex_20230221\dataset_graph_other']

    # 神经网络
    # model_nn = r'F:\U-net-train-val-test\model_trained\model_5_0.010342932306230068_UNet_u_net_block_train_loss.pth'
    model_nn = r'F:\U-net-train-val-test\model_trained\model_15_0.0011388693237677217_UNet_u_net_att_train_loss_66_150_1686207992.2917483.pth'

    # 输入参数
    # points_ploygon_input_1 = [[0, 5000], [10000, 5000], [10000, 0], [30000, 0], [30000, 20000], [0, 30000]]

    # name_project = 'typical-1'
    # points_ploygon_input_1 = [[0, 0], [22000, 0], [22000, 14000], [0, 14000],]
    # apartments_need_1 = {110: 2}
    # num_elevators_1 = 2

    #
    # name_project = 'typical-2'
    # points_ploygon_input_1 = [[0, 0], [34000, 0], [34000, 16000], [0, 16000], ]
    # apartments_need_1 = {90: 2, 80:2}
    # num_elevators_1 = 2
    # #
    #
    # name_project = 'typical-3'
    # points_ploygon_input_1 = [[0, 0], [38000, 0], [38000, 20000], [0, 20000], ]
    # apartments_need_1 = {100: 2, 60: 4}
    # num_elevators_1 = 2

    # name_project = 'untypical-1'
    # points_ploygon_input_1 = [[0, 0], [28000, 0], [28000, 8000], [18000, 8000],[18000, 14000],[10000, 14000],[10000, 8000], [0,8000] ]
    # apartments_need_1 = {120: 2, }
    # num_elevators_1 = 2

    # name_project = 'untypical-2'
    # width=10000
    # length=20000
    # points_ploygon_input_1 = [[length/2,0],[length/2, width],[-length/2, width],[-length/2,0],
    #                           [0,0],
    #                           [0,-width],[length,-width],[length,0],[length/2,0]]
    #
    # points_ploygon_input_1 = np.array(points_ploygon_input_1).reshape(-1, 2) \
    #                          - np.array([[length/4, 0]])\
    #                          + np.array([[256 / 2 * utils.size_grid, 256 / 2 * utils.size_grid]])
    #
    # plt.plot(np.array(points_ploygon_input_1).reshape(-1, 2)[:, 0],
    #          np.array(points_ploygon_input_1).reshape(-1, 2)[:, 1])
    # plt.show()
    # apartments_need_1 = {100: 2}
    # num_elevators_1 = 2

    name_project = 'untypical-3'
    height = 16000
    width = 10000
    points_ploygon_input_1 = [[width / 2 + height, 0], [width / 2 + height, width / 2], [width / 2, width / 2],
                              [width / 2, width / 2 + height],

                              [-width / 2, width / 2 + height], [-width / 2, width / 2],
                              [-width / 2 - height, width / 2], [-width / 2 - height, 0],

                              [-width / 2 - height, -width / 2], [-width / 2, -width / 2],
                              [-width / 2, -width / 2 - height], [0, -width / 2 - height],

                              [width / 2, -width / 2 - height], [width / 2, -width / 2, ],
                              [width / 2 + height, -width / 2], [width / 2 + height, 0]
                              ]
    # plt.Polygon(points_ploygon_input_1)
    points_ploygon_input_1 = np.array(points_ploygon_input_1).reshape(-1, 2) + np.array(
        [[256 / 2 * utils.size_grid, 256 / 2 * utils.size_grid]])
    plt.plot(np.array(points_ploygon_input_1).reshape(-1, 2)[:, 0],
             np.array(points_ploygon_input_1).reshape(-1, 2)[:, 1])
    plt.show()

    apartments_need_1 = {120: 4, }
    num_elevators_1 = 2

    # # 项目名称
    # name_project = '110_2_el_2'

    # Data 保存位置
    path_data_save = r'C:\Users\Administrator\Desktop\singleBuilding_unet_graph\Data_temp'

    path_project = os.path.join(path_data_save, name_project)
    name_files = ['Boundary', 'Marks_txt', 'Marks_pkl', 'Segment', 'Lines']
    for file in name_files:
        path_txt_save = os.path.join(path_project, file)
        if os.path.exists(path_txt_save):
            shutil.rmtree(path_txt_save)
        os.makedirs(path_txt_save)

    # 1.1 保存边界
    points_new_in_pic = stage_1_step_1_boundary_output.from_input_get_pix(points_ploygon_input_1, apartments_need_1)
    with open(os.path.join(os.path.join(path_project, name_files[0]), 'boundary.txt'), 'w') as f:
        for index in range(len(points_new_in_pic)):
            f.write(str(points_new_in_pic[index][0]))
            f.write('\n')
            f.write(str(points_new_in_pic[index][1]))
            f.write('\n')

        f.write(str(points_new_in_pic[0][0]))
        f.write('\n')
        f.write(str(points_new_in_pic[0][1]))
        f.write('\n')

    # 1.2 保存 marker txt
    boundary_input, list_markers = stage_1_step_4_generate_input.get_mark_nodes(points_ploygon_input_1,
                                                                                apartments_need_1,
                                                                                num_elevators_1,
                                                                                paths_png_test_1,
                                                                                paths_graph_test_1)

    for one, dict_loc in enumerate(list_markers):
        name_mark_ = 'mark_' + str(one) + '.txt'
        with open(os.path.join(os.path.join(path_project, name_files[1]), name_mark_), 'w') as f:
            for key, value in dict_loc.items():
                f.write(str(key))
                f.write('\n')
                f.write(str(value[0] * utils.size_grid))
                f.write('\n')
                f.write(str(value[1] * utils.size_grid))
                f.write('\n')

    # 1.3 保存输入网络的数据 marker pkl
    img_boundary = boundary_input[:, :, 0]
    img_region = boundary_input[:, :, 1]

    list_pkl_to_input = []
    for index, markers in enumerate(list_markers):
        composite = np.zeros((7, utils.size_pix, utils.size_pix))
        composite[0] = img_region
        composite[1] = img_boundary

        for key, value in markers.items():
            index_category = key // 100
            h, w = value
            min_h = max(h - utils.mask_size, 0)  # 找一下边界
            max_h = min(h + utils.mask_size, utils.size_pix - 1)
            min_w = max(w - utils.mask_size, 0)
            max_w = min(w + utils.mask_size, utils.size_pix - 1)

            composite[utils.num_info_boundary + index_category, min_h:max_h + 1, min_w:max_w + 1] = 1.0

        composite[2] = composite[utils.num_info_boundary:, :, :].sum(0)

        list_pkl_to_input.append(composite)

        print(composite)

    path_pkl_to_input = os.path.join(path_project, name_files[2])  # 文件地址
    for index_file, array_generate in enumerate(list_pkl_to_input):
        tensor_generate = torch.from_numpy(array_generate.astype(np.float32))
        pkl_name = os.path.join(path_pkl_to_input, str(index_file) + '.pkl')
        pkl_file = open(pkl_name, 'wb')
        pickle.dump(tensor_generate, pkl_file, protocol=pickle.HIGHEST_PROTOCOL)
        pkl_file.close()

    # 2 神经网络的分割
    path_image_output = os.path.join(path_project, name_files[3])
    generate_synth.synth(model_nn, path_pkl_to_input, path_image_output)

    # 3 find counter
    path_txt_line = os.path.join(path_project, name_files[4])
    for png in os.listdir(path_image_output)[0:]:
        path_img = os.path.join(path_image_output, png)
        list_largest_boundary, list_rect, list_lines = find_region.from_png_2_polygon(path_img)
        txt_name = png.split('.')[0]
        find_region.write_gemo_txt(list_largest_boundary, list_rect, list_lines,
                                   os.path.join(path_txt_line, txt_name + '.txt'))

    # rhino
