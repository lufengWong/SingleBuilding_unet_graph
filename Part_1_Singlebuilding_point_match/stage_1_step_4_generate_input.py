# -*- coding: utf-8 -*-
# @Time    : 2023/3/18 11:44
# @Author  : Lufeng Wang
# @WeChat  : tofind404
# @File    : stage_1_step_4_generate_input.py
# @Software: PyCharm

import os.path
import pickle

import numpy as np
import torch

import utils
from stage_1_step_3_fine_tune_3 import get_mark_nodes




if __name__ == '__main__':
    print('12')

    # 输入的信息 ######################
    # [0, 30000]
    points_ploygon_input_test_1 = [[0, 5000], [10000, 5000], [10000, 0], [30000, 0], [30000, 20000], [0, 30000]]
    apartments_need_test_1 = {100: 2, 80: 2, 65: 2}
    num_elevators_test_1 = 3

    # 平面图库
    paths_png_test_1 = [r'F:\Dataset_zjkj_4_channel_graph\building_piex_20230221\dataset_png_other_clear',
                        r'F:\Dataset_zjkj_4_channel_graph\building_piex_20230221\dataset_png_zjkj_clear']

    # 图结构库
    paths_graph_test_1 = [r'F:\Dataset_zjkj_4_channel_graph\building_piex_20230221\dataset_graph_zjkj',
                          r'F:\Dataset_zjkj_4_channel_graph\building_piex_20230221\dataset_graph_other']

    path_pkl_to_input = r'F:\U-net-train-val-test\pkl_to_input'

    # ################


    # ###############
    boundary_input, list_markers = get_mark_nodes(points_ploygon_input_test_1,
                                                  apartments_need_test_1,
                                                  num_elevators_test_1,
                                                  paths_png_test_1,
                                                  paths_graph_test_1)

    # print(boundary_input)
    # print(list_markers)
    # print('debut')
    #
    # path_txt = r'C:\Users\Administrator\Desktop\singleBuilding_unet_graph\Part_3_SingleBuilding_findCounter\txt_region_points'
    # for one, dict_loc in enumerate(list_markers):
    #     name_mark_ = 'mark_'+str(one)+'.txt'
    #     with open(os.path.join(path_txt, name_mark_), 'w') as f:
    #         for key, value in dict_loc.items():
    #             f.write(str(key))
    #             f.write('\n')
    #             f.write(str(value[0] * utils.size_grid))
    #             f.write('\n')
    #             f.write(str(value[1] * utils.size_grid))
    #             f.write('\n')


    # ################
    img_boundary = boundary_input[:, :, 0]
    img_region = boundary_input[:, :, 1]

    list_pkl_to_input = []
    for index, markers in enumerate(list_markers):
        composite =np.zeros((7, utils.size_pix, utils.size_pix))
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

    for index_file, array_generate in enumerate(list_pkl_to_input):
        tensor_generate = torch.from_numpy(array_generate.astype(np.float32))
        pkl_name = os.path.join(path_pkl_to_input, str(index_file)+'.pkl')
        pkl_file = open(pkl_name, 'wb')
        pickle.dump(tensor_generate, pkl_file, protocol=pickle.HIGHEST_PROTOCOL)
        pkl_file.close()






