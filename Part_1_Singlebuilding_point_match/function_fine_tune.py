import numpy as np

import networkx as nx
import matplotlib.pyplot as plt


def get_points_in(point_start, point_end):
    """
    生成直线段中的坐标点，包含前边不包含后边
    :param point_start:
    :param point_end:
    :return:
    """
    if point_start[0] == point_end[0]:
        x = point_start[0]
        if point_start[1] > point_end[1]:
            reverse = -1
        else:
            reverse = 1
        list_y = list(range(point_start[1], point_end[1], reverse))
        list_points = [[x, y_] for y_ in list_y]
        return list_points

    elif point_start[1] == point_end[1]:
        y = point_start[1]
        if point_start[0] > point_end[0]:
            reverse = -1
        else:
            reverse = 1
        list_x = list(range(point_start[0], point_end[0], reverse))
        list_points = [[x_, y] for x_ in list_x]
        return list_points

    else:
        return None


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
    assert len(neighbors) == mask_size ** 2, 'ROI 区域获取失败！'
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

    plt.figure(figsize=(8, 8))

    nx.draw_networkx_nodes(G, pos, node_size=600)
    nx.draw_networkx_edges(G, pos, width=[float(d['weight'] ** 0.5 * 0.1) for (u, v, d) in G.edges(data=True)])
    nx.draw_networkx_labels(G,
                            pos,
                            font_size=20,
                            font_family="sans-serif")
    # plt.axis("on")
    plt.grid()  # show grid
    plt.show()


if __name__ == '__main__':
    nodes_pos_debut = {0: (4, 4), 1: (1, 4), 2: (2, 6), 3: (1, 2), 4: (6, 2), 5: (6, 6)}
    nodes_area = {0: 20, 1: 10, 2: 5, 3: 5, 4: 80, 5: 80}
    nodes_adjacent = [[0, 1], [0, 2], [0, 3], [0, 4], [0, 5], [1, 2], [1, 3], [4, 5]]
    nodes_move = [2]
    assert len(nodes_move) == 1, '每次只一个，通过迭代实现所有，排在后边的可以先不放进去'
    points_ploygon = [[0, 1], [2, 1], [2, 0], [8, 0], [8, 8], [0, 8]]
    roi_mask_size = 5

    weight_boundary = nodes_area[nodes_move[0]] ** 2
    k_reject = 0.5

    # 固定的点
    nodes_fixed = [x for x in list(nodes_pos_debut.keys()) if x not in nodes_move]
    # 带权重的边
    nodes_weight_space = [(i, j, (nodes_area[i] ** 1) * (nodes_area[j] ** 1)) for [i, j] in nodes_adjacent]
    # 生成边界
    list_boundary_pos_group = [
        get_points_in(points_ploygon[index], points_ploygon[index + 1]) if index < len(points_ploygon) - 1
        else get_points_in(points_ploygon[index], points_ploygon[0])
        for index in range(len(points_ploygon))]

    # 每个点对应的感兴趣区域
    list_boundary_all = sum(list_boundary_pos_group, [])  # ROI 取点哪个添加哪个对应的感兴趣区域7x7
    pix_save = sum([get_ROI(nodes_pos_debut[node], roi_mask_size) for node in nodes_move], [])
    list_boundary_pos = [point for point in list_boundary_all if point in pix_save]

    print(list_boundary_pos)

    # 边界的定位转换成点
    num_space = max(list(nodes_pos_debut.keys())) + 1

    list_nodes_boundary = list(range(num_space, num_space + len(list_boundary_pos)))
    nodes_pos_debut.update(list(zip(list_nodes_boundary, list_boundary_pos)))

    nodes_weight_boundary_space = [(i, j, weight_boundary) for i in list_nodes_boundary for j in range(0, num_space)]

    # 综合
    nodes_weight_space += nodes_weight_boundary_space
    pos_fix = nodes_fixed + list_nodes_boundary
    fine_tune_one(nodes_weight_space, nodes_pos_debut, pos_fix, 1, k_reject)
