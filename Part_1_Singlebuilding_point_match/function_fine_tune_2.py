import copy

import numpy as np

import networkx as nx
import matplotlib.pyplot as plt

from shapely.geometry import MultiLineString, mapping, MultiPoint
from shapely.geometry import LineString, Polygon, Point, MultiPolygon, MultiPoint
from shapely.geometry import Point
from shapely.ops import split
#
# def get_points_in(point_start, point_end):
#     """
#     生成直线段中的坐标点，包含前边不包含后边
#     :param point_start:
#     :param point_end:
#     :return:
#     """
#     if point_start[0] == point_end[0]:
#         x = point_start[0]
#         if point_start[1] > point_end[1]:
#             reverse = -1
#         else:
#             reverse = 1
#         list_y = list(range(point_start[1], point_end[1], reverse))
#         list_points = [[x, y_] for y_ in list_y]
#         return list_points
#
#     elif point_start[1] == point_end[1]:
#         y = point_start[1]
#         if point_start[0] > point_end[0]:
#             reverse = -1
#         else:
#             reverse = 1
#         list_x = list(range(point_start[0], point_end[0], reverse))
#         list_points = [[x_, y] for x_ in list_x]
#         return list_points
#
#     else:
#         return None

# Python3 program to find an integer point
# on a line segment with given two ends
# function to find gcd of two numbers

def gcd(a, b):
    """
    ChatGPT NB
    GCD是英文greatest common divisor的缩写，意思是最大公约数。
    也就是说，两个或多个数能够整除的最大的正整数。
    例如，15和10的最大公约数是5，因为它们都能被5整除。15/5 = 3，10/5 = 2。
    :param a:
    :param b:
    :return:
    """
    if b == 0:
        return a
    return gcd(b, a % b)


# function to find an integer point on
# the line segment joining pointU and pointV
def findPoint(pointU, pointV):
    # ChatGPT NB
    # get x and y coordinates of points U and V
    # start end. has order
    x1 = pointU[0]
    y1 = pointU[1]
    x2 = pointV[0]
    y2 = pointV[1]

    # get absolute values of x and y differences
    dx = abs(x2 - x1)
    dy = abs(y2 - y1)

    # calculate gcd of dx and dy
    g = gcd(dx, dy)

    # if both differences are 0 then there is no other
    # valid integer point on the line segment other than (x1,y1)
    if g == 0:
        # print("(", x1, ",", y1, "), ", end="")
        return [[x1, y1]]

    # calculate increments in x and y coordinates for each possible solution
    incX = (x2 - x1) // g
    incY = (y2 - y1) // g

    # starting from first possible solution which is closest to U,
    # print points till last possible solution which is closest to V

    # for i in range(g + 1):
    #     print("(", (x1 + i * incX), ",", (y1 + i * incY), "), ", end="")

    return [[x1 + i * incX, y1 + i * incY] for i in range(g + 1)]


def get_ROI(point_target, mask_size):
    """
    获取周围正方形的index
    :param point_target: 目标点
    :param size_mask: ROI的区域
    :param pic: 边界图
    :return:
    """
    n = int((mask_size - 1) / 2)
    [i, j] = point_target
    neighbors = [[a, b] for a in range(i - n, i + n + 1) for b in range(j - n, j + n + 1)]
    # assert len(neighbors) == mask_size ** 2, 'ROI 区域获取失败！'
    return neighbors


def fine_tune_one(weighted_edges, pos_debut, pos_fiexed, scale_value, k, ):
    """

    :param k: Increase this value to move nodes farther apart.
    :param weighted_edges: [(0, 1, 0.5), (0, 2, 0.8), (1, 3, 0.7), (2, 4, 0.6), (3, 4, 0.9)]
    :param pos_debut:  {0: (0.5, 0.5), 1: (1.5, 1.5), 2: (2.5, -0.5), 3: (-1.5, -1.5), 4: (-2.5, -2.5)}  # initial positions
    :param pos_fiexed:  [3]
    :param scale_value:   0.8
    :return:
    """
    G = nx.Graph(graph_type='undirected')
    G.add_weighted_edges_from(weighted_edges)
    pos = pos_debut
    fixed = pos_fiexed  # fixed node
    scale = scale_value  # scale factor

    pos = nx.spring_layout(G, pos=pos, fixed=fixed, k=k, scale=scale, iterations=50)  # optimized positions

    # 移动的点
    nodes_move = [x for x in list(pos_debut.keys()) if x not in pos_fiexed]

    # 找出来相关的边
    list_edge = []
    for edge in G.edges:
        if nodes_move[0] in edge:
            list_edge.append(edge)

    # # 绘图 引力图
    # plt.figure(figsize=(8, 8))
    # nx.draw_networkx_nodes(G, pos, node_size=600)
    # nx.draw_networkx_edges(G,
    #                        pos=pos,
    #                        edgelist=list_edge ,
    #                        width =[float(d['weight'] ** 0.5 * 0.1) for (u, v, d) in G.edges(data=True)]
    #                        )
    # nx.draw_networkx_labels(G,
    #                         pos,
    #                         font_size=10,
    #                         font_family="sans-serif")
    # # plt.axis("on")
    # plt.grid()  # show grid
    # plt.title('Graph of gravity')
    # plt.show()

    return pos


def from_input_get_one_finetune(points_ploygon, nodes_pos_debut, nodes_area, nodes_adjacent, node_move,
                                weight_boundary=None,
                                roi_mask_size=5,
                                k_reject=0.5):
    """

    :param points_ploygon: 外轮廓的点
    :param nodes_pos_debut: 初始node的位置
    :param nodes_area: 每个node代表区域的面积
    :param nodes_adjacent: node的邻接关系
    :param node_move: 需要移动的点
    :param roi_mask_size: 感兴趣区域的大小
    :param weight_boundary: 引力边的权重
    :param k_reject: 斥力的权重
    :return:
    """

    # plt.show()

    # 为了适应原来的书写格式
    nodes_move = [node_move]

    # 使用经过调试的自适应weight
    if not weight_boundary:
        weight_boundary = nodes_area[node_move] ** 2

    # 固定的点
    nodes_fixed = [x for x in list(nodes_pos_debut.keys()) if x not in nodes_move]
    # 带权重的边
    nodes_weight_space = [(i, j, (nodes_area[i] ** 1) * (nodes_area[j] ** 1)) for [i, j] in nodes_adjacent]
    # 生成边界
    list_boundary_pos_group = [
        findPoint(points_ploygon[index], points_ploygon[index + 1]) if index < len(points_ploygon) - 1
        else findPoint(points_ploygon[index], points_ploygon[0])
        for index in range(len(points_ploygon))]

    # 每个点对应的感兴趣区域
    list_boundary_all = sum(list_boundary_pos_group, [])  # ROI 取点哪个添加哪个对应的感兴趣区域7x7
    pix_save = sum([get_ROI(nodes_pos_debut[node], roi_mask_size) for node in nodes_move], [])
    list_boundary_pos = [point for point in list_boundary_all if point in pix_save]

    # 边界的定位转换成点
    num_space = max(list(nodes_pos_debut.keys())) + 1

    list_nodes_boundary = list(range(num_space, num_space + len(list_boundary_pos)))
    nodes_pos_debut_boundary = copy.deepcopy(nodes_pos_debut)
    nodes_pos_debut_boundary.update(list(zip(list_nodes_boundary, list_boundary_pos)))

    nodes_weight_boundary_space = [(i, j, weight_boundary) for i in list_nodes_boundary for j in range(0, num_space)]

    # 综合
    nodes_weight_space += nodes_weight_boundary_space
    pos_fix = nodes_fixed + list_nodes_boundary
    position = fine_tune_one(nodes_weight_space, nodes_pos_debut_boundary, pos_fix, 1, k_reject)

    position_dic = {key: list(map(int, list(np.round(value, decimals=0)))) for key, value in position.items()}

    # 只保存原来的点
    position_dic_need = {key:position_dic[key] for key, value in  nodes_pos_debut.items()}

    return position_dic_need


if __name__ == '__main__':
    points_ploygon_1 = [[0, 1], [2, 1], [2, 0], [8, 0], [8, 8], [0, 8]]

    nodes_pos_debut_1 = {0: (4, 4), 1: (1, 4), 2: (2, 6), 3: (1, 2), 4: (6, 2), 5: (6, 6)}
    nodes_area_1 = {0: 20, 1: 10, 2: 5, 3: 5, 4: 80, 5: 80}
    nodes_adjacent_1 = [[0, 1], [0, 2], [0, 3], [0, 4], [0, 5], [1, 2], [1, 3], [4, 5]]
    nodes_move_1 = 2

    position_new = from_input_get_one_finetune(points_ploygon_1,
                                               nodes_pos_debut_1,
                                               nodes_area_1,
                                               nodes_adjacent_1,
                                               nodes_move_1)
    print(position_new)

    for key, value in position_new.items():
        plt.scatter(value[1], value[0])  # 为了和show_array 匹配
        plt.text(value[1], value[0], str(key))  # 为了和show_array 匹配
    plt.show()

    # # 固定的点
    # nodes_fixed = [x for x in list(nodes_pos_debut.keys()) if x not in nodes_move]
    # # 带权重的边
    # nodes_weight_space = [(i, j, (nodes_area[i] ** 1) * (nodes_area[j] ** 1)) for [i, j] in nodes_adjacent]
    # # 生成边界
    # list_boundary_pos_group = [
    #     findPoint(points_ploygon[index], points_ploygon[index + 1]) if index < len(points_ploygon) - 1
    #     else findPoint(points_ploygon[index], points_ploygon[0])
    #     for index in range(len(points_ploygon))]
    #
    # # 每个点对应的感兴趣区域
    # list_boundary_all = sum(list_boundary_pos_group, [])  # ROI 取点哪个添加哪个对应的感兴趣区域7x7
    # pix_save = sum([get_ROI(nodes_pos_debut[node], roi_mask_size) for node in nodes_move], [])
    # list_boundary_pos = [point for point in list_boundary_all if point in pix_save]
    #
    # print(list_boundary_pos)
    #
    # # 边界的定位转换成点
    # num_space = max(list(nodes_pos_debut.keys())) + 1
    #
    # list_nodes_boundary = list(range(num_space, num_space + len(list_boundary_pos)))
    # nodes_pos_debut.update(list(zip(list_nodes_boundary, list_boundary_pos)))
    #
    # nodes_weight_boundary_space = [(i, j, weight_boundary) for i in list_nodes_boundary for j in range(0, num_space)]
    #
    # # 综合
    # nodes_weight_space += nodes_weight_boundary_space
    # pos_fix = nodes_fixed + list_nodes_boundary
    # position = fine_tune_one(nodes_weight_space, nodes_pos_debut, pos_fix, 1, k_reject)
    # print('23')
