# -*- coding: utf-8 -*-
# @Time    : 2023/3/31 16:26
# @Author  : Lufeng Wang
# @WeChat  : tofind404
# @File    : ex2.py
# @Software: PyCharm
my_list = ['a', 'b', '', 'c', '', 'd', 'e', []]
# my_list = list(filter(lambda x: x != '', my_list))
# print(my_list)

lst = list(filter(None, my_list))
print(lst)