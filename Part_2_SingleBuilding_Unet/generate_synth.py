import os
import pickle

import numpy as np
from network_candidates import unet_attention_building_spatialAttention_outline_pooling_blockForm as unet
# from network_candidates import unet_building as unet
import torch

from torchvision.utils import save_image
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


def synth(model, path_image_input, path_image_output):
    # 使用的设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 网络
    net = unet.UNet().to(device)
    net.load_state_dict(torch.load(model))

    for pkl_pth in os.listdir(path_image_input):
        path_pkl = os.path.join(path_image_input, pkl_pth)
        f = open(path_pkl, 'rb')
        tensor_composite = pickle.load(f)
        tensor_composite = tensor_composite.to(device).view(1, 7, 256, 256)
        print(tensor_composite.shape)

        out = net(tensor_composite)
        print('shape out')
        print(out.shape)
        x = (tensor_composite[0, 0, :, :] + tensor_composite[0, 2, :, :]).view(1, 256, 256)
        print('shape x')
        print(x.size())
        y_ = out[0][0].view(1, 256, 256)
        print('shape y_')
        print(y_.size())
        image = torch.stack([x, y_], 0)
        print('shape image')
        print(image.size())

        # plt.show()
        save_image(image.cpu(), os.path.join(path_image_output, f"{pkl_pth.split('.')[0]}.png"), padding=0)


if __name__ == '__main__':
    print('12')
    model = r'F:\U-net-train-val-test\model_trained\model_47_0.0017480459064245224_UNet_u_net_baseline.pth'
    path_image_input = r'F:\U-net-train-val-test\test_pkl_had\synth_input_pkl_input'
    # path_image_input = r'F:\pkl_to_input'  # 自己绘制的
    path_image_output = r'F:\U-net-train-val-test\test_pkl_had\synth_input_Pkl_output'

    synth(model, path_image_input, path_image_output)
