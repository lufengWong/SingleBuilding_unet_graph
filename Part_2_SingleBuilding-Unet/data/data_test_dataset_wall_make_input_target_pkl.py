import shutil

from function_floorplan_train import LoadFloorplanTrain
from torch.utils import data
import torch as t
import random
import utils
import os
import pickle
import numpy as np
import PIL.Image
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

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
    patches = [mpatches.Patch(color=colors[i], label="Label {l}".format(l=int(values[i]))) for i in range(len(values))]
    # put those patched as legend-handles into the legend
    plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

    # plt.grid(True)
    plt.title(name)


class WallDataset(data.Dataset):
    def __init__(self, data_root, mask_size):
        self.mask_size = mask_size
        self.floorplans = [os.path.join(data_root, pth_path) for pth_path in os.listdir(data_root)]

    def __len__(self):
        return len(self.floorplans)

    def __getitem__(self, index):
        floorplan_path = self.floorplans[index]
        floorplan = LoadFloorplanTrain(floorplan_path, self.mask_size)  # 实例化一个函数

        OUTSIDE = 2
        NOTHING = 1
        INTERIORWALL = 0

        # 添加标记点
        living_node = floorplan.living_node
        # print(living_node)
        floorplan.add_room(living_node)

        continue_node = floorplan.continue_node
        for node in continue_node:  # 会对node的种类进行判断
            floorplan.add_room(node)

        # 输入的信息就是边界和区域
        input = floorplan.get_composite_wall(num_extra_channels=0)  # 输入数据

        # 输出的信息中有内墙
        target = t.zeros((floorplan.data_size, floorplan.data_size), dtype=t.long)  # 输出数据
        target[target == 0] = 0  # 默认是2  #
        # target[floorplan.inside != 0] = NOTHING  # 内部为1
        target[floorplan.interiorWall == 1] = 1 # 内部的墙为0
        # target[floorplan.interiordoor == 1] = INTERIORWALL  # 内部的门为0
        return input, target


if __name__ == '__main__':

    path_pkl_debut = r'F:\dataset_pkl\exam_pkl\all'
    path_pkl_input = r'F:\dataset_pkl\exam_pkl\synth_input_pkl_input'
    path_pkl_target = r'F:\dataset_pkl\exam_pkl\synth_input_pkl_target'
    mask_size = 2  # 三是合适得

    if os.path.exists(path_pkl_input):
        shutil.rmtree(path_pkl_input)
    os.mkdir(path_pkl_input)

    if os.path.exists(path_pkl_target):
        shutil.rmtree(path_pkl_target)
    os.mkdir(path_pkl_target)


    data_train = list(WallDataset(path_pkl_debut, mask_size))
    floor_plans_names = [pth_path for pth_path in os.listdir(path_pkl_debut)]

    for data, name in zip(data_train, floor_plans_names):
        # print(data[0].shape)

        # show_array(data[0][1, :, :] + data[0][2, :, :], '1')
        # plt.show()

        # for i in range(7):
        #     show_array(data[0][i, :, :], str(i))
        #     plt.show()

        print('1')
        path_name_pkl_new = os.path.join(path_pkl_input, 'input_'+str(name))
        pkl_file = open(path_name_pkl_new, 'wb')

        path_name_pkl_new2 = os.path.join(path_pkl_target, 'target_' + str(name))
        pkl_file2 = open(path_name_pkl_new2, 'wb')

        # 保存了 内部区域的标注，外部墙的标记，内部墙的标记，字典-房间的质点  # 其实只有这些
        pickle.dump(data[0], pkl_file, protocol=pickle.HIGHEST_PROTOCOL)
        pkl_file.close()

        pickle.dump(data[1], pkl_file2, protocol=pickle.HIGHEST_PROTOCOL)
        pkl_file.close()

    # for pth_path in os.listdir(path_pkl_debut):
    #     # floor_plan = os.path.join(path_pkl_debut, pth_path)
    #     train_data = WallDataset(data_root=path_pkl_debut, pth_path=pth_path, mask_size=mask_size)
    #     print('12')
    #     print(train_data)
    #     # [input, target] = list(train_data)
    #     k = list(train_data)

    #
    # print(list(train_data))
    # print(len(list(train_data)))
    # # print('---------------')
    # # print(input)
    # # print('---------------')
    # # print(target)
    # path_name_pkl_new = os.path.join(path_pkl_input, pth_path)
    #
    # # 上述两步骤操作只是创建了新的文件
    # pkl_file = open(path_name_pkl_new, 'wb')
    # # 保存了 内部区域的标注，外部墙的标记，内部墙的标记，字典-房间的质点  # 其实只有这些
    # pickle.dump(list(train_data)[0], pkl_file, protocol=pickle.HIGHEST_PROTOCOL)
    # pkl_file.close()
