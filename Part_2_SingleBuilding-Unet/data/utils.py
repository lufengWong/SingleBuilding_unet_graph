# room_label = [(0, 'LivingRoom'),
#               (1, 'MasterRoom'),
#               (2, 'Kitchen'),
#               (3, 'Bathroom'),
#               (4, 'DiningRoom'),
#               (5, 'ChildRoom'),
#               (6, 'StudyRoom'),
#               (7, 'SecondRoom'),
#               (8, 'GuestRoom'),
#               (9, 'Balcony'),
#               (10, 'Entrance'),
#               (11, 'Storage'),
#               (12, 'Wall-in'),
#               (13, 'External'),
#               (14, 'ExteriorWall'),
#               (15, 'FrontDoor'),
#               (16, 'InteriorWall'),
#               (17, 'InteriorDoor')]
size_pix = 256
size_grid = 192
value_step = 100

room_label = [
              (0, 'Apartment'),
              (1, 'Ladder'),
              (2, 'Lift'),
              (3, 'Public'),

              (4, 'External'),
              (5, 'ExteriorWall'),
              (6, 'InteriorWall'),
              ]



# 只要了0~12的类别
category = [category for category in room_label if category[1] not in {'External', 'ExteriorWall',
                                                                       'InteriorWall'}]

num_category = len(category)   # 数据长度
num_info_boundary = 3  # boundary inside sum_category

# pixel2length = 18 / 256


def label2name(label=0):
    if label < 0 or label > len(room_label):
        raise Exception("Invalid label!", label)
    else:
        return room_label[label][1]


def label2index(label=0):
    if label < 0 or label > len(room_label):
        raise Exception("Invalid label!", label)
    else:
        return label


def compute_centroid(mask):
    sum_h = 0
    sum_w = 0
    count = 0
    shape_array = mask.shape
    for h in range(shape_array[0]):
        for w in range(shape_array[1]):
            if mask[h, w] != 0:
                sum_h += h
                sum_w += w
                count += 1

    return (sum_h // count, sum_w // count)


def log(file, msg='', is_print=True):
    if is_print:
        print(msg)
    file.write(msg + '\n')
    file.flush()
