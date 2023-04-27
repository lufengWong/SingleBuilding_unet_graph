# -*- coding: utf-8 -*-
# @Time    : 2023/4/11 21:36
# @Author  : Lufeng Wang
# @WeChat  : tofind404
# @File    : ex1.py
# @Software: PyCharm
import torch # 如果pytorch安装成功即可导入
print(torch.cuda.is_available()) # 查看CUDA是否可用
print(torch.cuda.device_count()) # 查看可用的CUDA数量
print(torch.version.cuda) # 查看CUDA的版本号

import torch
print(torch.__version__)

import numpy as np
from scipy.interpolate import interp1d

y = [[3],5,9,7,18,16,6,[5],9,10]
# x = np.linspace(0,9,num=10) # x = [0,1,2,3,4,5,6,7,8,9]
# f = interp1d(x,y) # 创建一个线性插值函数
# x_pred = np.linspace(0,9,num=10) # 在区间[0,9]内均匀取10个点
# y_pred = f(x_pred) # 用插值函数计算这些点对应的值
# print(y_pred) # 输出结果

y=y[::3]
print(y)