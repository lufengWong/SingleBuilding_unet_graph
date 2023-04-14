# -*- coding: utf-8 -*-
# @Time    : 2023/4/5 15:18
# @Author  : Lufeng Wang
# @WeChat  : tofind404
# @File    : paper_need.py
# @Software: PyCharm
import matplotlib.pyplot as plt
import networkx as nx
from PIL import Image

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import numpy as np
import cv2

import networkx as nx


def draw_graph(g_true):
    # 图结构可视化
    ID_COLOR = {1: '#EE4D4D', 2: '#C67C7B', 3: '#FFD274', 4: '#BEBEBE', 5: '#BFE3E8', 6: '#7BA779',
                7: '#E87A90', 8: '#FF8C69', 10: '#1F849B', 15: '#727171', 16: '#785A67', 17: '#D3A2C7'}

    # build true graph
    G_true = nx.Graph()
    colors_H = []
    node_size = []
    edge_color = []
    linewidths = []
    edgecolors = []

    # add nodes
    for k, label in enumerate(g_true[0]):
        _type = label + 1
        if _type >= 0:
            G_true.add_nodes_from([(k, {'label': k + 1})])
            colors_H.append(ID_COLOR[_type])
            node_size.append(1000)
            edgecolors.append('blue')
            linewidths.append(0.0)
    # add edges
    for k, m, l in g_true[1]:
        _type_k = g_true[0][k] + 1
        _type_l = g_true[0][l] + 1
        if m > 0:
            G_true.add_edges_from([(k, l)])
            edge_color.append('#D3A2C7')

    #     # # visualization - debug
    #     print(len(node_size))
    #     print(len(colors_H))
    #     print(len(linewidths))
    #     print(G_true.nodes())
    #     print(g_true[0])
    #     print(len(edgecolors))
    # 画图
    plt.figure()
    pos = nx.nx_agraph.graphviz_layout(G_true, prog='neato')
    nx.draw(G_true, pos, node_size=node_size, linewidths=linewidths, node_color=colors_H, font_size=14,
            font_color='white', \
            font_weight='bold', edgecolors=edgecolors, edge_color=edge_color, width=4.0, with_labels=False)
    node_labels = nx.get_node_attributes(G_true, 'label')
    nx.draw_networkx_labels(G_true, pos, labels=node_labels)
    plt.tight_layout()
    # plt.savefig('./dump/_true_graph.jpg', format="jpg")
    # plt.close('all')
    # rgb_im = Image.open('../dump/_true_graph.jpg')
    # return G_true, rgb_im


def show_array(array_img, name):
    """
    矩阵， 图像
    :param array_img:
    :param name:
    :return:
    """
    # im = plt.imshow(array_img, cmap='Pastel1')
    im = plt.imshow(array_img, cmap='rainbow')
    values = np.unique(array_img.ravel())
    # get the colors of the values, according to the
    # colormap used by imshow
    colors = [im.cmap(im.norm(value)) for value in values]
    # create a patch (proxy artist) for every color
    patches = [mpatches.Patch(color=colors[i], label="Mask {l}".format(l=int(values[i])))
               for i in range(len(values))]
    # put those patched as legend-handles into the legend
    plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., fontsize=25)  # 24

    # plt.grid(True)
    plt.title(name, font={'family': 'Arial', 'size': 40})  # 20


if __name__ == '__main__':
    print('12')

    path_img = r'C:\Users\Administrator\Desktop\singleBuilding_unet_graph\1_1_1.png'
    path_graph = r'F:\Dataset_zjkj_4_channel_graph\CAD_layout\example\dataset_graph_zjkj\1_1_1.npy'

    # img = np.array(Image.open(path_img))
    # print(img)
    # show_array(img[:, :, 3], 'Channel 4 : Inside mask')
    # plt.show()

    graph_tuple = np.load(path_graph, allow_pickle=True)
    print(graph_tuple)
    # draw_graph(path_graph)
    #
    # import networkx as nx
    # import matplotlib.pyplot as plt
    #
    G = nx.Graph()
    # data = [[101, 101], [101, 301], [101, 6], [201, 201], [201, 301], [201, 2], [301, 101], [301, 201], [301, 301],
    #         [301, 1], [301, 2], [301, 3], [301, 4], [301, 5], [301, 6], [1, 301], [1, 1], [1, 2], [2, 201], [2, 301],
    #         [2, 1], [2, 2], [3, 301], [3, 3], [3, 4], [4, 301], [4, 3], [4, 4], [5, 301], [5, 5], [5, 6], [6, 101],
    #         [6, 301], [6, 5], [6, 6]]
    #
    data = [one for one in graph_tuple if one[0] != one[1]]
    G.add_edges_from(data)
    pos = nx.spring_layout(G)  # 可以选择其他布局算法
    nx.draw_networkx(G, pos, node_size=1000, node_color='dodgerblue', font_size=16)
    plt.grid()
    plt.title('Graph')
    plt.show()
