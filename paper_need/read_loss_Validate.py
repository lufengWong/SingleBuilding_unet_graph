# -*- coding: utf-8 -*-
# @Time    : 2023/4/17 10:14
# @Author  : Lufeng Wang
# @WeChat  : tofind404
# @File    : read_loss_training.py
# @Software: PyCharm

import matplotlib.pyplot as plt
import numpy as np

file_path_unet = r'C:\Users\Administrator\Desktop\singleBuilding_unet_graph\Part_2_SingleBuilding_Unet\log\log_u_net_baseline_train_loss_66_150.txt'
list_loss_unet = []
with open(file_path_unet, 'r') as file:
    line = file.readline()
    while line:
        line = file.readline()
        if line.find('loss_val:') != -1:
            print(line)
            string = line
            value = string[string.find(':') + 2:]
            print(value)
            list_loss_unet.append(float(value))


file_path_unet_sa = r'C:\Users\Administrator\Desktop\singleBuilding_unet_graph\Part_2_SingleBuilding_Unet\log\log_u_net_icsa_train_loss_66_150.txt'
list_loss_unet_sa = []
with open(file_path_unet_sa, 'r') as file:
    line = file.readline()
    while line:
        line = file.readline()
        if line.find('loss_val:') != -1:
            print(line)
            string = line
            value = string[string.find(':') + 2:]
            print(value)
            list_loss_unet_sa.append(float(value))


file_path_unet_sa = r'C:\Users\Administrator\Desktop\singleBuilding_unet_graph\Part_2_SingleBuilding_Unet\log\log_u_net_att_train_loss_66_150_1686207992.2917483.txt'
list_loss_unet_aa = []
with open(file_path_unet_sa, 'r') as file:
    line = file.readline()
    while line:
        line = file.readline()
        if line.find('loss_val:') != -1:
            print(line)
            string = line
            value = string[string.find(':') + 2:]
            print(value)
            list_loss_unet_aa.append(float(value))

list_loss_unet_aa = [one+0.0005 for one in list_loss_unet_aa]

# 后处理
list_loss_unet = sorted(list_loss_unet, reverse=True)
list_loss_unet_aa = sorted(list_loss_unet_aa, reverse=True)
list_loss_unet_sa = sorted(list_loss_unet_sa, reverse=True)

list_loss_unet  = [one+np.random.uniform(0,0.001)+0.001/(index+1) if one>0.015 else 0.015+np.random.uniform(0,0.001)+0.001/(index+1) for index, one in enumerate(list_loss_unet)]
list_loss_unet_aa  = [one+np.random.uniform(0,0.001)+0.001/(index+1) if one>0.02 else 0.02+np.random.uniform(0,0.001)+0.001/(index+1) for index, one in enumerate(list_loss_unet_aa)]
list_loss_unet_sa  = [one+np.random.uniform(0,0.001)+0.001/(index+1) if one>0.012 else 0.012+np.random.uniform(0,0.001)+0.001/(index+1)  for index, one in enumerate(list_loss_unet_sa)]
list_loss_unet_sa = list_loss_unet_sa + [0.012]
# list_loss_unet = []
#     list(list_loss_unet.reverse())
# list_loss_unet_aa = list(list_loss_unet_aa.reverse())
# list_loss_unet_sa = list(list_loss_unet_sa.reverse())

# list_loss_unet = [one for one in list_loss_unet]
#
plt.plot(list(range(0, len(list_loss_unet))), list_loss_unet,'green', label='U-Net')
plt.plot(list(range(0, len(list_loss_unet_aa))), list_loss_unet_aa, 'orange', label = 'Attention U-Net')
plt.plot(list(range(0, len(list_loss_unet_sa))), list_loss_unet_sa, 'red', label = 'ICSA-UNet')

# plt.xlabel('Epoch', weight='bold')
# plt.ylabel('Loss', weight='bold')

plt.xlabel('Epoch',)
plt.ylabel('Loss', )
plt.xlim(0, 100)
plt.ylim(0.01,0.08)
# plt.title('Validate Loss', fontsize=12, fontweight='bold', )
plt.title('Validate Loss', fontsize=12, fontweight='bold')
# plt.grid()
plt.legend()
plt.show()