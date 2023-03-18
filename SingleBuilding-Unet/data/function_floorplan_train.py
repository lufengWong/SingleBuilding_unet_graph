from PIL import Image
import numpy as np
import torch as t
import random
import pickle
import utils


class LoadFloorplanTrain():
    """
    Loading a floorplan for train
    这种函数 全部作为属性了，已经作为方法，只有需要的进行一个输出
    """

    def __init__(self, floorplan_path, mask_size, random_shuffle=True):
        "Read data from pickle"
        with open(floorplan_path, 'rb') as pkl_file:
            [inside, boundary, interiorWall, room_node] = pickle.load(
                pkl_file)  # 如果在原始的数据上使用一部分进行预训练，很难预测所有的墙，所以只需要一部分对应的内墙

        "randomly order rooms"  # 重排列房间的{category, (x,y)}
        if random_shuffle:
            random.shuffle(room_node)

        self.boundary = t.from_numpy(boundary)  # 把数组转换成张量，且二者共享内存，对张量进行修改比如重新赋值，那么原始数组也会相应发生改变。
        self.interiorWall = t.from_numpy(interiorWall)
        # self.interiordoor = t.from_numpy(interiordoor)
        self.inside = t.from_numpy(inside)
        self.data_size = self.inside.shape[0]  # 数据的格式
        self.mask_size = mask_size  # 扩大点的大小 ###################### 此时才有这回是

        self.continue_node = []
        for node in room_node:  # 做了一个分类
            if node['category'] == 3:  # 先统计类别 # 原模型中living只可能有一个，现在是public3 ##########################
                self.living_node = node
            else:
                self.continue_node.append(node)

        # 做掩码
        "inside_mask"
        self.inside_mask = t.zeros((self.data_size, self.data_size))
        self.inside_mask[self.inside != 0] = 1.0  # (0,1)  # 内部为1

        "boundary_mask"
        self.boundary_mask = t.zeros((self.data_size, self.data_size))
        self.boundary_mask[self.boundary == 127] = 1.0
        # self.boundary_mask[self.boundary == 255] = 0.5

        "front_door_mask"
        # self.front_door_mask = t.zeros((self.data_size, self.data_size))
        # self.front_door_mask[self.boundary == 255] = 1.0

        "category_mask"  # 先造了个空的矩阵
        self.category_mask = t.zeros(
            (utils.num_category, self.data_size, self.data_size))  # 只有12个！！！！！！！！！没有做区分的 卧室和卧室之间
        # 后边会有添加 add room

    def get_composite_wall(self, num_extra_channels=0):  #
        composite = t.zeros(
            (utils.num_category + num_extra_channels + utils.num_info_boundary, self.data_size, self.data_size))
        composite[0] = self.inside_mask
        composite[1] = self.boundary_mask
        # composite[2] = self.front_door_mask
        composite[2] = self.category_mask.sum(0)  # 每个种类的标记点
        for i in range(utils.num_category):  # 这里限制到只要前几项
            composite[i + utils.num_info_boundary] = self.category_mask[i]  # 第i个维度的
        return composite  # 4+12=16 dimension

    def add_room(self, room):
        """
        添加一个质点的矩形空间
        Args:
            room:
        Returns:
        """

        index = utils.label2index(room['category'])  # label就是编号，确保在编号符合规范  # 这里会添加所有的类别，但是没有公共区真的很尴尬
        h, w = room['centroid']
        min_h = max(h - self.mask_size, 0)  # 找一下边界
        max_h = min(h + self.mask_size, self.data_size - 1)
        min_w = max(w - self.mask_size, 0)
        max_w = min(w + self.mask_size, self.data_size - 1)
        self.category_mask[index, min_h:max_h + 1, min_w:max_w + 1] = 1.0  # 不断对不同的维度进行填充
