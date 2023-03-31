# -*- coding: utf-8 -*-

import copy
#
#import utils

import rhinoscriptsyntax as rs
import scriptcontext as sc



def get_seg_rect_line_list(path_boundary, path_segment):


    # draw boundary ############################################

    # 打开文件
    fileHandler = open(path_boundary, "r")
    # 获取文件中所有行的列表
    listOfLines = fileHandler.readlines()
    # 关闭文件
    fileHandler.close()
    # 遍历列表中的每一行

    list_outline = []

    count = -1
    point_this = [0, 0, 0]
    for line in listOfLines:
        count += 1
        value = int(line.strip())
        if count % 2 == 0:
            point_this[1] = value
        else:
            point_this[0] = value
            list_outline.append(point_this)
            point_this = [0, 0, 0]

    # draw segment ############################################
    # 打开文件
    fileHandler = open(path_segment, "r")
    # 获取文件中所有行的列表
    listOfLines = fileHandler.readlines()
    # 关闭文件
    fileHandler.close()
    # 遍历列表中的每一行

    list_boundary = []
    list_rects = []
    list_lines = []

    mark_type = None
    list_points = []

    for line in listOfLines:
        print(line.strip())
        if line.strip() == 'Boundary':
            if mark_type == 0:
                list_boundary.append(list_points)
            elif mark_type == 1:
                list_rects.append(list_points)
            else:
                list_lines.append(list_points)

            mark_type = 0
            list_points = []

        elif line.strip() == 'Rect':
            if mark_type == 0:
                list_boundary.append(list_points)
            elif mark_type == 1:
                list_rects.append(list_points)
            else:
                list_lines.append(list_points)

            mark_type = 1
            list_points = []

        elif line.strip() == 'Line':
            if mark_type == 0:
                list_boundary.append(list_points)
            elif mark_type == 1:
                list_rects.append(list_points)
            else:
                list_lines.append(list_points)

            mark_type = 2
            list_points = []

        elif line.strip() == 'over':
            if mark_type == 0:
                list_boundary.append(list_points)
            elif mark_type == 1:
                list_rects.append(list_points)
            else:
                list_lines.append(list_points)

        else:
            value = int(line.strip())
            print(value)
            list_points.append(value)

    # print('debut')

    ####################################
    # boundary - segment
    list_segment_points = []
    for seg in list_boundary:  # 遍历每一个线
        list_seg_this = []
        point_this = [0, 0, 0]
        for index, value in enumerate(seg):
            value = value
            if index % 2 == 0:
                point_this[0] = value
            else:
                point_this[1] = value
                list_seg_this.append(point_this)
                point_this = [0, 0, 0]
        list_segment_points.append(list_seg_this)

    # rect
    list_rect_points = []
    for seg in list_rects:  # 遍历每一个线
        list_seg_this = []
        point_this = [0, 0, 0]
        for index, value in enumerate(seg):
            value = value
            if index % 2 == 0:
                point_this[0] = value
            else:
                point_this[1] = value
                list_seg_this.append(point_this)
                point_this = [0, 0, 0]
        list_rect_points.append(list_seg_this)

    # line
    list_line_points = []
    for seg in list_lines:  # 遍历每一个线
        list_seg_this = []
        point_this = [0, 0, 0]
        for index, value in enumerate(seg):
            value = value
            if index % 2 == 0:
                point_this[0] = value
            else:
                point_this[1] = value
                list_seg_this.append(point_this)
                point_this = [0, 0, 0]
        list_line_points.append(list_seg_this)

    list_outline = list(filter(None, list_outline))
    list_segment_points = list(filter(None, list_segment_points))
    list_rect_points = list(filter(None, list_rect_points))
    list_line_points = list(filter(None, list_line_points))

    print(list_outline)
    print(list_segment_points)
    print(list_rect_points)
    print(list_line_points)


    return list_outline, list_segment_points, list_rect_points, list_line_points


if __name__ == "__main__":
    path_boundary_1 = r'C:\Users\Administrator\Desktop\singleBuilding_unet_graph\data_gererate\boundary.txt'
    path_segment_1 = r'C:\Users\Administrator\Desktop\singleBuilding_unet_graph\Part_3_SingleBuilding_findCounter\txt_region_points\2.txt'


    list_outline, list_segment_points, list_rect_points, list_line_points = \
        get_seg_rect_line_list(path_boundary_1, path_segment_1,)

    print(12)

    # 获取场景中的所有对象
    objects = rs.AllObjects()
    # 删除所有对象
    rs.DeleteObjects(objects)
    # 将单位设置为毫米
    rs.UnitSystem(2)

    # 绘制边界轮廓 ##########################
    print(list_outline)
    # 添加多边形
    polyline = rs.AddPolyline(list_outline)
    # 闭合多边形
    rs.CloseCurve(polyline)
    # 创建多边形表面并添加到场景中
    srf = rs.AddPlanarSrf(polyline)
#    rs.AddSurface(srf)

    # 向上拉伸多线段
    height = 1000  # 拉伸高度
    extrude_srf = rs.ExtrudeCurveStraight(polyline, (0, 0, 0), (0, 0, height))

    # 绘制大分割线 ##########################
    print(list_segment_points)
    for seg in list_segment_points:
        # 添加多边形
        polyline = rs.AddPolyline(seg)

    # 绘制小分割线 ##########################
    print(list_line_points)
    for seg in list_line_points:
        # 添加多边形
        polyline = rs.AddPolyline(seg)

    # # 绘制筒 ##########################
    print(list_rect_points)
    for seg in list_rect_points:
        # 添加多边形
        polyline = rs.AddPolyline(seg)
        # 闭合多边形
        rs.CloseCurve(polyline)

        # 向上拉伸多线段
        height = 3000  # 拉伸高度
        extrude_srf = rs.ExtrudeCurveStraight(polyline, (0, 0, 0), (0, 0, height))
        # 获取封闭线段的边界框
        bbox = rs.BoundingBox(extrude_srf)

        # 添加边界框
        rs.AddBox(bbox)
# #







# list_boundary.append(list_points)
#
#     list_points = []
# else:
#     list_points.append(int(line.strip()))

# list_points_all.append(list_points)
#
# list_points_all = list_points_all[1:]
# list_polygon_all = []
# for points_list in list_points_all:
#     list_polygon_this = []
#     point_this = [0, 0, 0]
#     for index, value in enumerate(points_list):
#
#         if index % 2 == 0:
#             point_this[1] = value
#         else:
#             point_this[0] = value
#             list_polygon_this.append(point_this)
#             point_this = [0, 0, 0]
#
#     list_polygon_all.append(list_polygon_this)
#


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


# # draw segment ############################################
# # 打开文件
# fileHandler = open (path_segment, "r")
# # 获取文件中所有行的列表
# listOfLines = fileHandler.readlines()
# # 关闭文件
# fileHandler.close()
# # 遍历列表中的每一行
# list_points_all =[]
# list_points = []
# for line in listOfLines:
#     if line.strip() == 'Points':
#         list_points_all.append(list_points)
#         list_points = []
#     else:
#         list_points.append(int(line.strip()))
#
# list_points_all.append(list_points)
#
# list_points_all = list_points_all[1:]
# list_polygon_all = []
# for points_list in list_points_all:
#     list_polygon_this = []
#     point_this = [0, 0, 0]
#     for index, value in enumerate(points_list):
#
#         if index % 2 == 0:
#             point_this[0] = value * size_grid
#         else:
#             point_this[1] = value * size_grid
#             list_polygon_this.append(point_this)
#             point_this = [0, 0, 0]
#
#     list_polygon_all.append(list_polygon_this)
#
# list_points_all_height = []
# for points in copy.deepcopy(list_polygon_all):
#     list_points_all_height_this = []
#     for point in points[::-1]:
#         point[2] = 3000
#         list_points_all_height_this.append(point)
#     list_points_all_height.append(list_points_all_height_this)
#
# print('000000000000')
# list_polygon_all_last = []
# for list_ground, list_height in zip(list_polygon_all, list_points_all_height):
#     print('66666666666')
#     print(list_ground)
#     print(list_height)
#     for point in list_height:
#         print('------')
#         list_ground.append(point)
#     list_ground.append(list_ground[0])
#
#     print('-----------------')
#     print(list_ground)
#     list_polygon_all_last.append(list_ground)
#
# print(list_polygon_all_last)
#
# for points in list_polygon_all_last:
#     polygon = rs.AddPolyline(points)
#
#     # 将多边形转换为闭合的曲线
#     closed_curve = rs.CloseCurve(polygon)
#
#     surface_id = rs.AddPlanarSrf(closed_curve)


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
