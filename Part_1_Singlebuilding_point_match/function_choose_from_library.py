import numpy as np
from scipy.spatial.distance import euclidean


def get_apartment_elevator_similar(num_elevator,
                                   list_target,
                                   list_num_elevators,
                                   list_candidates_area, num_choose=10):
    """
    获得与目标相似的户型
    :param num_choose: 选择几个面积相似的
    :param num_elevator: 电梯个数
    :param list_target: 需要的户型的面积
    :param list_num_elevators: 备选的户型的梯户比
    :param list_candidates: 备选的户型的面积
    :return: 选出来的10个面积相似的户型
    """

    # step 1 楼梯和房间的个数相同
    # step 2 房间的面积尽量相同 进行排序而后欧式距离取前10
    list_index_euclidean = [[index, euclidean(sorted(list_area), sorted(list_target))]
                            for index, (num_apart, list_area) in
                            enumerate(zip(list_num_elevators, list_candidates_area))
                            if len(list_area) == len(list_target) and num_apart == num_elevator]

    # 将a和b打包成元组，并按照a的值进行排序
    list_indexes_elevator_same = [one[0] for one in list_index_euclidean]
    list_length_euclidean = [one[1] for one in list_index_euclidean]

    sorted_pairs = sorted(zip(list_length_euclidean, list_indexes_elevator_same))
    # 解包排序后的元组
    sorted_similarity_length, sorted_similarity_index = zip(*sorted_pairs)
    chosen_buildings_index = list(map(int, sorted_similarity_index[0:int(num_choose)]))

    return chosen_buildings_index


def get_region_similar(region_target, list_regions_candidations, num_chose_to_generate=5):
    """
    比较边界的几何相似性 旋转的是户型库里边的户型
    :param region_target: 二维数组
    :param list_regions_candidations: 二维数组列表
    :param num_chose_to_generate: 选择个数进行下一步的生成
    :return: 户型库对应的户型及其编码
    """

    def dice(array_a, array_b):
        """ 实为1 """
        dice_calculate = (2 * np.sum(array_a * array_b)) / (np.sum(array_a) + np.sum(array_b))

        return dice_calculate

    # 需要旋转自己的图和图库进行比较得到一个最大的Dice值
    def max_dice(array_input, array_one_in_library):
        """
        旋转的是 array_one_in_library
        :param array_input:
        :param array_one_in_library:
        :return:
        """
        list_reverse = [0, 1]  # 正反对称
        list_angles = [0, 1, 2, 3]  # 旋转的角度0 90 180 270

        list_array_re_rot = [np.rot90(np.fliplr(array_one_in_library), angle)
                             if mark_re == 1 else np.rot90(array_one_in_library, angle)
                             for angle in list_angles for mark_re in list_reverse]

        list_array_re_rot_index = [[mark_re, angle] for angle in list_angles for mark_re in list_reverse]

        list_dices = [dice(array_this, array_input) for array_this in list_array_re_rot]
        max_dice = max(list_dices)
        max_dice_index = list_dices.index(max(list_dices))  # 此处只返回一个最靠前的

        # 返回各个角度中最大的值，以及对应的角度
        return [max_dice, list_array_re_rot_index[max_dice_index]]

    # 获取此户型与户型库每一个的最大dice值
    list_dices_rot = [max_dice(np.array(region_target), np.array(array_choose)) for array_choose in list_regions_candidations]
    # 获得dice值以及旋转的角度
    list_dices = [dice_rot[0] for dice_rot in list_dices_rot]
    list_rots = [dice_rot[1] for dice_rot in list_dices_rot]  # re rot

    # 将a和b打包成元组，并按照a的值进行排序
    list_boundaries_chosen_index = list(range(len(list_regions_candidations)))
    sorted_pairs = sorted(zip(list_dices, list_boundaries_chosen_index), reverse=True)
    # 解包排序后的元组
    sorted_similarity_boundary_dice, sorted_similarity_boundary_index = zip(*sorted_pairs)
    chosen_buildings_index = list(map(int, sorted_similarity_boundary_index[0:int(num_chose_to_generate)]))
    # 获得对应的旋转角度
    chosen_buildings_rot = [list_rots[index] for index in chosen_buildings_index]

    return chosen_buildings_index, chosen_buildings_rot


if __name__ == '__main__':
    # test1
    num_elevators = 3
    list_target = [65, 67, 89]

    list_num_apart_an_elevator = [2, 3, 3, 2, 3]
    list_area_apart = [[70, 70, 70],
                       [65, 89, 67],
                       [65, 67, 99],
                       [65, 67, 99],
                       [65, 67, 89]]

    index_chosen = get_apartment_elevator_similar(num_elevators,
                                                  list_target,
                                                  list_num_apart_an_elevator,
                                                  list_area_apart)
    print(index_chosen)

    # test2
    region_target_1 = np.array([[1, 1], [0, 1]])
    list_boundaries_1 = list(
        [np.array([[1, 1], [0, 0]]),
         np.array([[0, 1], [1, 1]]),
         np.array([[1, 1], [0, 1]]),
         np.array([[1, 1], [1, 1]])]
    )

    get_like = get_region_similar(region_target_1, list_boundaries_1, 2)
    print(get_like)
