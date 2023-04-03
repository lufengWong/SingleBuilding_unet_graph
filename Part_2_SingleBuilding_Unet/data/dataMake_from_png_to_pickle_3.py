import copy
import copyreg
import math
import random

import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image
import numpy as np
import shutil
import pickle
import utils
import os

import sys

import matplotlib.patches as mpatches

import datetime
import multiprocessing as mp


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


def get_piex_outline(img_input, mark_target_exterior, mark_interior_wall, mark_other):
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
    img = copy.deepcopy(img_input).astype(np.uint8)
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
        kernel = np.ones((8, 8), np.uint8)
        dilation = cv2.dilate(img_out_copy, kernel, iterations=1)

        img_in = img - dilation
        img_in[img_in == -255] = 0
        img_in[img_in == 255] = mark_interior_wall

        return img_out.astype(np.uint8), img_in.astype(np.uint8)
    else:
        print("存在多条外围轮廓，请检查！")
        sys.exit()


def write2pickle(train_dir, pkl_dir):
    """
    并行处理pkl
    """
    # train_data_path = [os.path.join(train_dir, path) for path in os.listdir(train_dir)]

    train_data_names = [name for name in os.listdir(train_dir)]
    print(f'Number of dataset: {len(train_data_names)}')

    pool = mp.Pool(12)
    result = [pool.apply_async(write2pickle_single_pic_crop, args=(train_dir_1, pkl_dir_1, name_1, count_1))
              for train_dir_1, pkl_dir_1, name_1, count_1 in
              zip([train_dir] * (len(train_data_names)), [pkl_dir] * (len(train_data_names)), train_data_names,
                  range(len(train_data_names)))]

    [p.get() for p in result]


def write2pickle_single(train_dir, pkl_dir, name, count_num):
    """
    将一个png转化为需要的pkl
    :param train_dir:
    :param pkl_dir:
    :param name:
    :return:
    """
    print(count_num)

    path = os.path.join(train_dir, name)

    with Image.open(path) as temp:
        image_array = np.asarray(temp, dtype=np.uint8)

    boundary_mask = image_array[:, :, 0]  # 边界
    category_mask = image_array[:, :, 1]  # 类别
    index_mask = image_array[:, :, 2]  # 标记index
    inside_mask = image_array[:, :, 3]  # 区域

    shape_array = image_array.shape  # （256， 256， 4）直接在每个像素点
    index_category = []
    room_node = []

    # show_array(category_mask, 'category_mask')
    # plt.show()

    # 内部分隔墙 此时需要根据预训练采用的category进行进一步的确定########################
    interiorWall_mask = np.zeros(category_mask.shape, dtype=np.uint8)
    interiorWall_mask[category_mask == 6] = 1  # debut 16 to 6  预训练还是可以 其实还是16，不过还是要根据目标房间
    # show_array(interiorWall_mask, 'wall_interior')
    # plt.show()

    ####################################

    category_list = [0, 1, 2, 3]  # 根据预训练使用的图纸的位置选择，只要是三个通道就OK的 ######################################

    # 内部的门
    # interiordoor_mask = np.zeros(category_mask.shape, dtype=np.uint8)
    # interiordoor_mask[category_mask == 17] = 1

    # 所有的room区域，把类别和index区分，为了找质心
    for h in range(shape_array[0]):
        for w in range(shape_array[1]):
            index = index_mask[h, w]  # 遍历每一个语义分割后区域的像素点的index
            category = category_mask[h, w]  # 获得分类的标签category
            if index > 0 and (
                    category in category_list):  # 所有的 ==room区域== ，且类别为房间区域  # category < debut 12 ###############################
                if len(index_category):  # 如果已经存在数据对
                    flag = True

                    for i in index_category:
                        if i[0] == index and i[1] == category:  # 如果此区域的像素点的index 存在
                            flag = False  # 为false， 一直为false

                    if flag:  # 如果flag是Ture则进行存入
                        index_category.append((index, category))  # 是像素值 而不是 位置
                else:
                    index_category.append((index, category))


    for (index, category) in index_category:  # 遍历所有的实例分割后的标记号码以及对应的分类
        node = {}  # 可以有很多node
        node['category'] = int(category)  # 不同node的category可以相同
        mask = np.zeros(index_mask.shape, dtype=np.uint8)

        # add
        judge1 = index_mask == index
        judge2 = category_mask == category
        judge_intersect = judge1 & judge2

        mask[judge_intersect] = 1  # 所有实例分割后编号和所需要的编号的区域， 的像素值为1

        node['centroid'] = utils.compute_centroid(mask)  # 计算质心
        room_node.append(node)  # 添加质心的位置 {category:(x,y)}

    # print('----------')
    # print(room_node)

    list_room_nodes = [list(point['centroid']) for point in room_node]
    # print(list_room_nodes)
    room_node_mask = np.zeros(category_mask.shape, dtype=np.uint8)
    for point in list_room_nodes:
        room_node_mask[point[0], point[1]] = 255

    # show_array(room_node_mask, 'room_node')
    # plt.show()

    # 膨胀质点
    # kernel_dilate = np.ones((4, 4), np.uint8)
    # img_dilate = cv2.dilate(room_node_mask, kernel_dilate, iterations=1)
    # show_array(img_dilate, 'room_node_dilate')
    # plt.show()

    # 将再次处理后的信息存入pkl文件
    pkl_path = path.replace(train_dir, pkl_dir)  # 将原图片的路径改为pkl的路径
    pkl_path = pkl_path.replace('png', 'pkl')  # 将原图片的png改为pkl后缀
    # 上述两步骤操作只是创创建了新的文件
    pkl_file = open(pkl_path, 'wb')
    # 保存了 内部区域的标注，外部墙的标记，内部墙的标记，字典-房间的质点  # 其实只有这些
    pickle.dump([inside_mask, boundary_mask, interiorWall_mask, room_node], pkl_file, protocol=pickle.HIGHEST_PROTOCOL)
    pkl_file.close()

    return 0


def mask_rectangle_Cut_out(size_pic, mask_width_ratio=0.5, mask_height_ratio=0.5):
    """
    生成一个矩形遮挡
    """

    size_pic = size_pic - 1

    mask_width = math.ceil(size_pic * mask_width_ratio) # 取大
    mask_height = math.ceil(size_pic * mask_height_ratio)

    width_available = size_pic - mask_width
    height_available = size_pic - mask_height

    # 开始的位置
    point_width_index_start = math.ceil(random.uniform(0, 0.99) * width_available) # 取小
    point_height_index_start = math.ceil(random.uniform(0, 0.99) * height_available)

    points_mask = [[x, y]
                   for x in range(point_width_index_start, point_width_index_start + mask_width-1)
                   for y in range(point_height_index_start, point_height_index_start + mask_height-1)]

    judge = np.all(np.less(np.array(points_mask), 256))

    if not judge:
        print('111111111111111111111------------------')
        print(points_mask)
        print(mask_width)

    assert judge, '超出坐标轴'
    return points_mask


def make_mask(array_img, list_mask, prob=1, list_channel_fill=[0, 4, 0, 0]):
    """
    进行遮挡区域的替换
    """
    # 可能不遮挡
    if random.random() > prob:
        return array_img

    # < prob
    for index, fill in enumerate(list_channel_fill):
        for mask in list_mask:
            array_img[:, :, index][mask[0], mask[1]] = fill

    return array_img


def write2pickle_single_pic_crop(train_dir, pkl_dir, name, count_num):
    """
        将一个png转化为需要的多个crop的pkl
        :param train_dir:
        :param pkl_dir:
        :param name:
        :return:
        """
    print(count_num// (4+1), count_num % (4+1))

    path = os.path.join(train_dir, name)

    with Image.open(path) as temp:
        image_array = np.asarray(temp, dtype=np.uint8)

    list_image_arrays = [image_array]
    for i in range(4):
        points_mask = mask_rectangle_Cut_out(utils.size_pix, random.uniform(0, 0.5), random.uniform(0, 0.5))

        image_array_crop = make_mask(copy.deepcopy(image_array), points_mask, prob=0.5)
        list_image_arrays.append(image_array_crop)

    for index_crop, image_array in enumerate(list_image_arrays):  # 5 张
        boundary_mask = image_array[:, :, 0]  # 边界
        category_mask = image_array[:, :, 1]  # 类别
        index_mask = image_array[:, :, 2]  # 标记index
        inside_mask = image_array[:, :, 3]  # 区域

        shape_array = image_array.shape  # （256， 256， 4）直接在每个像素点
        index_category = []
        room_node = []

        # show_array(category_mask, 'category_mask')
        # plt.show()

        # 内部分隔墙 此时需要根据预训练采用的category进行进一步的确定########################
        interiorWall_mask = np.zeros(category_mask.shape, dtype=np.uint8)
        interiorWall_mask[category_mask == 6] = 1  # debut 16 to 6  预训练还是可以 其实还是16，不过还是要根据目标房间
        # show_array(interiorWall_mask, 'wall_interior')
        # plt.show()

        ####################################

        category_list = [0, 1, 2, 3]  # 根据预训练使用的图纸的位置选择，只要是三个通道就OK的 ######################################

        # 内部的门
        # interiordoor_mask = np.zeros(category_mask.shape, dtype=np.uint8)
        # interiordoor_mask[category_mask == 17] = 1

        # 所有的room区域，把类别和index区分，为了找质心
        for h in range(shape_array[0]):
            for w in range(shape_array[1]):
                index = index_mask[h, w]  # 遍历每一个语义分割后区域的像素点的index
                category = category_mask[h, w]  # 获得分类的标签category
                if index > 0 and (
                        category in category_list):  # 所有的 ==room区域== ，且类别为房间区域  # category < debut 12 ###############################
                    if len(index_category):  # 如果已经存在数据对
                        flag = True

                        for i in index_category:
                            if i[0] == index and i[1] == category:  # 如果此区域的像素点的index 存在
                                flag = False  # 为false， 一直为false

                        if flag:  # 如果flag是Ture则进行存入
                            index_category.append((index, category))  # 是像素值 而不是 位置
                    else:
                        index_category.append((index, category))


        for (index, category) in index_category:  # 遍历所有的实例分割后的标记号码以及对应的分类
            node = {}  # 可以有很多node
            node['category'] = int(category)  # 不同node的category可以相同
            mask = np.zeros(index_mask.shape, dtype=np.uint8)

            # add
            judge1 = index_mask == index
            judge2 = category_mask == category
            judge_intersect = judge1 & judge2

            mask[judge_intersect] = 1  # 所有实例分割后编号和所需要的编号的区域， 的像素值为1

            node['centroid'] = utils.compute_centroid(mask)  # 计算质心
            room_node.append(node)  # 添加质心的位置 {category:(x,y)}

        # print('----------')
        # print(room_node)

        list_room_nodes = [list(point['centroid']) for point in room_node]
        # print(list_room_nodes)
        room_node_mask = np.zeros(category_mask.shape, dtype=np.uint8)
        for point in list_room_nodes:
            room_node_mask[point[0], point[1]] = 255

        # show_array(room_node_mask, 'room_node')
        # plt.show()

        # 膨胀质点
        # kernel_dilate = np.ones((4, 4), np.uint8)
        # img_dilate = cv2.dilate(room_node_mask, kernel_dilate, iterations=1)
        # show_array(img_dilate, 'room_node_dilate')
        # plt.show()

        # 将再次处理后的信息存入pkl文件
        pkl_path = path.replace(train_dir, pkl_dir)  # 将原图片的路径改为pkl的路径
        pkl_path = pkl_path.replace('.png', '_'+str(index_crop + 1) + '.pkl')  # 将原图片的png改为pkl后缀
        # 上述两步骤操作只是创创建了新的文件
        pkl_file = open(pkl_path, 'wb')
        # 保存了 内部区域的标注，外部墙的标记，内部墙的标记，字典-房间的质点  # 其实只有这些
        pickle.dump([inside_mask, boundary_mask, interiorWall_mask, room_node], pkl_file,
                    protocol=pickle.HIGHEST_PROTOCOL)
        pkl_file.close()

    return 0


if __name__ == '__main__':
    # # Lufeng Wang ###################
    # train_dir =f'dataset\\png'
    # train_data_path = [os.path.join(train_dir, path) for path in os.listdir(train_dir)]
    # print(f'Number of dataset: {len(train_data_path)}')
    #
    # for path in train_data_path[0:2]:
    #     with Image.open(path) as temp:
    #         image_array = np.asarray(temp, dtype=np.uint8)
    #     boundary_mask = image_array[:, :, 0]
    #     category_mask = image_array[:, :, 1]
    #     index_mask = image_array[:, :, 2]
    #     inside_mask = image_array[:, :, 3]
    #     shape_array = image_array.shape

    # # print("*******************************************")
    # dataset_floorplan = r"F:\data_zjkj\dataset_png_other_clear"
    #
    # train_dataset_dir = r"F:\dataset_pkl\train_pic"
    train_dataset_dir = r'F:\dataset_U-net\train_pic'
    # val_dataset_dir = r"F:\dataset_pkl\val_pic"
    #
    # # 先分成两类，训练集和验证集
    # if os.path.exists(train_dataset_dir):
    #     shutil.rmtree(train_dataset_dir)
    # os.mkdir(train_dataset_dir)
    #
    # if os.path.exists(val_dataset_dir):
    #     shutil.rmtree(val_dataset_dir)
    # os.mkdir(val_dataset_dir)
    #
    # percent_train = 0.7
    #
    # list_pic = os.listdir(dataset_floorplan)
    # for pic in tqdm(list_pic):
    #     if random.random() <= percent_train :
    #         # print('train')
    #         shutil.copy(os.path.join(dataset_floorplan, pic), os.path.join(train_dataset_dir, pic))
    #     else:
    #         # print('valuate')
    #         shutil.copy(os.path.join(dataset_floorplan, pic), os.path.join(val_dataset_dir, pic))
    #
    # 进行数据的处理和保存
    # train_pickle_dir = r"F:\dataset_pkl\train"
    train_pickle_dir = r'F:\dataset_U-net\train_reinforce'
    # val_pickle_dir = r"F:\dataset_pkl\val"
    #
    # 如果存在处理后的数据则进行删除
    if os.path.exists(train_pickle_dir):
        shutil.rmtree(train_pickle_dir)
    os.mkdir(train_pickle_dir)
    # #
    # # if os.path.exists(val_pickle_dir):
    # #     shutil.rmtree(val_pickle_dir)
    # # os.mkdir(val_pickle_dir)
    #
    write2pickle(train_dataset_dir, train_pickle_dir)
    # write2pickle(val_dataset_dir, val_pickle_dir)

    # # 转化验证集 ####################################
    # dataset_dir_other = r'F:\data_zjkj\data_png_exam\other'
    # dataset_dir_zjkj = r'F:\data_zjkj\data_png_exam\zjkj'
    # pickle_dir = r'F:\dataset_pkl\exam_pkl\all'
    #
    # # if os.path.exists(pickle_dir):
    # #     shutil.rmtree(pickle_dir)
    # # os.mkdir(pickle_dir)
    #
    # write2pickle(dataset_dir_zjkj, pickle_dir)
