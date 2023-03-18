import pickle
import shutil

import numpy as np
from PIL import Image
import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


# 　删除重复的png

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


def similar(array_a, array_b):
    res = np.array(array_a == array_b)
    count_ture = np.sum(res == True)
    print(count_ture)
    num_ = array_a.shape[0] * array_a.shape[1]
    print(num_)
    percent_ture = count_ture / num_
    print(percent_ture)
    return percent_ture


if __name__ == '__main__':
    # 两部分对应清理重读  # 应该根据轮廓的相似度
    image_debut = r'F:\data_zjkj\dataset_png_zjkj'
    image_clear = r'F:\data_zjkj\dataset_png_zjkj_clear'

    if os.path.exists(image_clear):
        shutil.rmtree(image_clear)
    os.mkdir(image_clear)

    list_data_already = []
    for pth_path in os.listdir(image_debut):

        print('-----------------------------------------------')
        print(pth_path)

        path_png = os.path.join(image_debut, pth_path)
        path_png_clear = os.path.join(image_clear, pth_path)

        array_img = np.array(Image.open(path_png))

        # show_array(array_img[:,:,0], '12')
        # plt.show()
        if any(np.array_equal(array_img, i) for i in list_data_already):  # 存在相同
        # if any(similar(array_img[:,:,0], i[:,:,0]) > 0.99 for i in list_data_already): # 存在相同
            print('already being*************************************')
            pass
        else:
            list_data_already.append(array_img)
            shutil.copy(path_png, path_png_clear)

    # 两部分删除对比重复
    # # 删除other中的数据
    # image_zjkj = r'F:\data_zjkj\dataset_png_zjkj_clear'
    # image_other = r'F:\data_zjkj\dataset_png_other_clear'
    # list_data_already = [np.array(Image.open(os.path.join(image_zjkj, p))) for p in os.listdir(image_zjkj)]
    #
    # for index, pth_path in enumerate(os.listdir(image_other)):
    #     print('-----------------------------')
    #     print(index)
    #     print(pth_path)
    #
    #     path_png = os.path.join(image_other, pth_path)
    #
    #     array_img = np.array(Image.open(path_png))
    #
    #     if any(np.array_equal(array_img, i) for i in list_data_already):  # 存在相同
    #         print('already being***************** delete ********************')
    #         os.remove(path_png)

    # # 删除没有public的图片 有些可能是因为没有质心造成的
    # image_files = r'F:\data_zjkj\dataset_png_other_clear'
    # for pth_path in os.listdir(image_files):
    #     path_png = os.path.join(image_files, pth_path)
    #     array_img = np.array(Image.open(path_png))
    #
    #     # show_array(array_img[:, :, 1], 'public')
    #     # plt.show()
    #
    #     if array_img.__contains__(3) and np.sum(array_img==3) > 100: # 不准确 ######################
    #         print(np.sum(array_img==3))  # 不准确 ######################
    #         pass
    #     else:
    #         print('no public or small space***************** delete ********************')
    #         os.remove(path_png)
    #
    # 删除没有public的pkl
    # image_files = r'F:\dataset_pkl\exam_pkl\all'
    # for index,  pth_path in enumerate (os.listdir(image_files)):
    #     print('-----------------')
    #     print(index)
    #     path_png = os.path.join(image_files, pth_path)
    #
    #     with open(path_png, 'rb') as pkl_file:
    #         [inside, boundary, interiorWall, room_node] = pickle.load(pkl_file)
    #
    #     find_public = False
    #     for node in room_node:
    #         if node['category'] == 3:
    #             find_public = True
    #             break
    #
    #     if not find_public:
    #         print('no public or small space***************** delete ********************')
    #         os.remove(path_png)
