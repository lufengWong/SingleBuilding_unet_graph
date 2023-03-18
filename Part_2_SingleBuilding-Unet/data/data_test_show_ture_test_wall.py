import pickle
import shutil

from PIL import Image
import numpy as np
import os
import torch as t

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
    patches = [mpatches.Patch(color=colors[i], label="Pixel value {l}".format(l=int(values[i]))) for i in
               range(len(values))]
    # put those patched as legend-handles into the legend
    plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., fontsize=7)  # 24

    # plt.grid(True)
    plt.title(name, fontsize=10)  # 20



if __name__ == '__main__':

    path_pkl_target = r'C:\Users\Administrator\Desktop\SingleBuilding_piex2building_13\net_syn\synth_input_pkl_target'
    path_png_ture = r'C:\Users\Administrator\Desktop\SingleBuilding_piex2building_13\net_syn\synth_ture'

    if os.path.exists(path_png_ture):
        shutil.rmtree(path_png_ture)
    os.mkdir(path_png_ture)

    for pth_path in os.listdir(path_pkl_target):
        print(pth_path)
        path = os.path.join(path_pkl_target, pth_path)
        with open(path, 'rb') as pkl_file:
            tensor_ture = pickle.load(pkl_file)
            name_pic = os.path.join(path_png_ture, pth_path.replace('pkl', 'png'))
            show_array(tensor_ture, 'ture')
            plt.savefig(name_pic)
            plt.show()


    # data_train = list(WallDataset(path_pkl_debut, mask_size))
    # floor_plans_names = [pth_path for pth_path in os.listdir(path_pkl_debut)]
    #
    # for data, name in zip(data_train, floor_plans_names):
    #     print('1')
    #     path_name_pkl_new = os.path.join(path_pkl_input, 'input_' + str(name))
    #     pkl_file = open(path_name_pkl_new, 'wb')
    #
    #     path_name_pkl_new2 = os.path.join(path_pkl_target, 'target_' + str(name))
    #     pkl_file2 = open(path_name_pkl_new2, 'wb')
    #
    #     # 保存了 内部区域的标注，外部墙的标记，内部墙的标记，字典-房间的质点  # 其实只有这些
    #     pickle.dump(data[0], pkl_file, protocol=pickle.HIGHEST_PROTOCOL)
    #     pkl_file.close()
    #
    #     pickle.dump(data[1], pkl_file2, protocol=pickle.HIGHEST_PROTOCOL)
    #     pkl_file.close()
    #
    # output = Image.fromarray(np.uint8(self.map[:, :, 1]))  # 只返回了这一层， 修改一下
    #
    #
