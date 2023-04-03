# -*- coding: utf-8 -*-

# import copy
# #
# #import utils
#
# import rhinoscriptsyntax as rs
# import scriptcontext as sc
#
#
# # 获取场景中的所有对象
# objects = rs.AllObjects()
#
# # 删除所有对象
# rs.DeleteObjects(objects)
#
# # 将单位设置为毫米
# rs.UnitSystem(2)


path_boundary = r'C:\Users\Administrator\Desktop\singleBuilding_unet_graph\data_gererate\boundary.txt'
path_segment = r'C:\Users\Administrator\Desktop\singleBuilding_unet_graph\Part_3_SingleBuilding_findCounter\txt_region_points\0.txt'

size_grid = 192

# draw boundary ############################################
# 打开文件
fileHandler = open (path_boundary, "r")
# 获取文件中所有行的列表
listOfLines = fileHandler.readlines()
# 关闭文件
fileHandler.close()
# 遍历列表中的每一行
list_points_all =[]
list_points = []
for line in listOfLines:
    print(line.strip())
    if line.strip() == 'Points':
        list_points_all.append(list_points)
        list_points = []
    else:
        list_points.append(int(line.strip()))

list_points_all.append(list_points)

list_points_all = list_points_all[1:]
list_polygon_all = []
for points_list in list_points_all:
    list_polygon_this = []
    point_this = [0, 0, 0]
    for index, value in enumerate(points_list):

        if index % 2 == 0:
            point_this[1] = value
        else:
            point_this[0] = value
            list_polygon_this.append(point_this)
            point_this = [0, 0, 0]

    list_polygon_all.append(list_polygon_this)

# for points in list_polygon_all:
#
#     print(points)
#
#
#     polygon = rs.AddPolyline(points)
#
#     # 将多边形转换为闭合的曲线
#     closed_curve = rs.CloseCurve(polygon)
#
#     surface_id = rs.AddPlanarSrf(closed_curve)

#    material_index = rs.AddMaterial("Red")
#    rs.ObjectMaterialIndex(polygon, material_index)

#   # 添加一个材质
#   material_index = sc.doc.Materials.Add()
#   sc.doc.Materials[material_index].DiffuseColor = (255, 0, 0)
#   rs.ObjectMaterialSource(surface_id, 1)
#   rs.ObjectMaterialIndex(surface_id, material_index)



# draw segment ############################################
# 打开文件
fileHandler = open (path_segment, "r")
# 获取文件中所有行的列表
listOfLines = fileHandler.readlines()
# 关闭文件
fileHandler.close()
# 遍历列表中的每一行
list_points_all =[]
list_points = []
for line in listOfLines:
    if line.strip() == 'Points':
        list_points_all.append(list_points)
        list_points = []
    else:
        list_points.append(int(line.strip()))

list_points_all.append(list_points)

list_points_all = list_points_all[1:]
list_polygon_all = []
for points_list in list_points_all:
    list_polygon_this = []
    point_this = [0, 0, 0]
    for index, value in enumerate(points_list):

        if index % 2 == 0:
            point_this[0] = value * size_grid
        else:
            point_this[1] = value * size_grid
            list_polygon_this.append(point_this)
            point_this = [0, 0, 0]

    list_polygon_all.append(list_polygon_this)

list_points_all_height = []
for points in copy.deepcopy(list_polygon_all):
    list_points_all_height_this = []
    for point in points[::-1]:
        point[2] = 3000
        list_points_all_height_this.append(point)
    list_points_all_height.append(list_points_all_height_this)

print('000000000000')
list_polygon_all_last = []
for list_ground, list_height in zip(list_polygon_all, list_points_all_height):
    print('66666666666')
    print(list_ground)
    print(list_height)
    for point in list_height:
        print('------')
        list_ground.append(point)
    list_ground.append(list_ground[0])

    print('-----------------')
    print(list_ground)
    list_polygon_all_last.append(list_ground)

print(list_polygon_all_last)

for points in list_polygon_all_last:
    polygon = rs.AddPolyline(points)

    # 将多边形转换为闭合的曲线
    closed_curve = rs.CloseCurve(polygon)

    surface_id = rs.AddPlanarSrf(closed_curve)


    # print(points)
#
#     polygon = rs.AddPolyline(points)
#
#     # 将多边形转换为闭合的曲线
#     closed_curve = rs.CloseCurve(polygon)
#
#     surface_id = rs.AddPlanarSrf(closed_curve)
#
# #    # 添加一个材质
# #    material_index = sc.doc.Materials.Add()
# #    sc.doc.Materials[material_index].DiffuseColor = (255, 0, 0)
# #    rs.ObjectMaterialSource(surface_id, 1)
# #    rs.ObjectMaterialIndex(surface_id, material_index)
#
#
#
#
# # 更新场景
# sc.doc.Views.Redraw()



