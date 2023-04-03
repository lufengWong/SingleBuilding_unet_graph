# -*- coding: utf-8 -*-
# @Time    : 2023/4/3 14:49
# @Author  : Lufeng Wang
# @WeChat  : tofind404
# @File    : main_generate.py
# @Software: PyCharm
import os

import utils
from Part_1_Singlebuilding_point_unet_align import stage_1_step_1_boundary_output
from Part_1_Singlebuilding_point_unet_align import stage_1_step_4_generate_input

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

    # 输入参数
    points_ploygon_input_1 = [[0, 5000], [10000, 5000], [10000, 0], [30000, 0], [30000, 20000], [0, 20000]]
    apartments_need_1 = {100: 2, 80: 2, 65: 1}
    num_elevators_1 = 3

    # txt 保存位置
    path_txt_save = r'C:\Users\Administrator\Desktop\singleBuilding_unet_graph\Part_4_Rhino_data_gererate\txt_region_points'

    # 保存边界
    points_new_in_pic = stage_1_step_1_boundary_output.from_input_get_pix(points_ploygon_input_1, apartments_need_1)
    with open(os.path.join(path_txt_save, 'boundary.txt'), 'w') as f:
        for index in range(len(points_new_in_pic)):
            f.write(str(points_new_in_pic[index][0]))
            f.write('\n')
            f.write(str(points_new_in_pic[index][1]))
            f.write('\n')

        f.write(str(points_new_in_pic[0][0]))
        f.write('\n')
        f.write(str(points_new_in_pic[0][1]))
        f.write('\n')

    # 保存 marker
    boundary_input, list_markers =  stage_1_step_4_generate_input.get_mark_nodes(points_ploygon_input_1,
                                                      apartments_need_1,
                                                      num_elevators_1,
                                                      paths_png_test_1,
                                                      paths_graph_test_1)

    print(boundary_input)
    print(list_markers)
    print('debut')

    for one, dict_loc in enumerate(list_markers):
        name_mark_ = 'mark_' + str(one) + '.txt'
        with open(os.path.join(path_txt_save, name_mark_), 'w') as f:
            for key, value in dict_loc.items():
                f.write(str(key))
                f.write('\n')
                f.write(str(value[0] * utils.size_grid))
                f.write('\n')
                f.write(str(value[1] * utils.size_grid))
                f.write('\n')




