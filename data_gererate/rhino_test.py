# -*- coding: utf-8 -*-
# @Time    : 2023/3/31 15:04
# @Author  : Lufeng Wang
# @WeChat  : tofind404
# @File    : rhino_test.py
# @Software: PyCharm
import rhinoscriptsyntax as rs

# 获取场景中的所有对象
objects = rs.AllObjects()

# 删除所有对象
rs.DeleteObjects(objects)


pts = [(0,0,0), (0,10,0), (10,10,0), (10,0,0)]
poly = rs.AddPolyline(pts)
rs.CloseCurve(poly)
height = 10 # 设置拉伸高度
cap = False # 是否封顶
rs.ExtrudeCurveStraight(poly, (0,0,0),(0,0,height))
