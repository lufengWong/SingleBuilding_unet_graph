from .function_floorplan_train import LoadFloorplanTrain
from torch.utils import data
import torch as t
import random
import utils
import os


class WallDataset(data.Dataset):
    """
    使用floor_plan_train生成最后的输入数据256*256和输出数据256*256
    """
    def __init__(self, data_root, mask_size):
        self.mask_size = mask_size
        self.floorplans = [os.path.join(data_root, pth_path) for pth_path in os.listdir(data_root)]

    def __len__(self):
        return len(self.floorplans)

    def __getitem__(self, index):
        floorplan_path = self.floorplans[index]
        floorplan = LoadFloorplanTrain(floorplan_path, self.mask_size)  # 实例化一个函数

        OUTSIDE = 0
        NOTHING = 2
        INTERIORWALL = 1

        # 添加标记点  ######## living 不需要这么特殊
        # living_node = floorplan.living_node
        # floorplan.add_room(living_node) # add

        continue_node = floorplan.continue_node
        for node in continue_node:  # 会对node的种类进行判断
            floorplan.add_room(node)

        # 最终的网络输入和输出
        # 输入的信息就是边界和区域 ####################################
        input = floorplan.get_composite_wall(num_extra_channels=0)  # 输入数据

        # 输出的信息中有内墙 # 其实就一层？ ##################################
        target = t.zeros((floorplan.data_size, floorplan.data_size), dtype=t.long)  # 输出数据
        target[target == 0] = 0  # 默认是2  # #################
        # target[floorplan.inside != 0] = NOTHING  # 内部为1
        target[floorplan.interiorWall == 1] = 1  # 内部的墙为0
        # target[floorplan.interiordoor == 1] = INTERIORWALL  # 内部的门为0

        # print('target--------------')
        # print(target.shape)
        target = target.view(1, 256, 256)
        return input, target


