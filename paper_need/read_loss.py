# -*- coding: utf-8 -*-
# @Time    : 2023/4/17 10:14
# @Author  : Lufeng Wang
# @WeChat  : tofind404
# @File    : read_loss.py
# @Software: PyCharm

import matplotlib.pyplot as plt

file_path_unet = r'C:\Users\Administrator\Desktop\singleBuilding_unet_graph\Part_2_SingleBuilding_Unet\log\log_u_net_baseline_train_loss.txt'
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


file_path_unet_sa = r'C:\Users\Administrator\Desktop\singleBuilding_unet_graph\Part_2_SingleBuilding_Unet\log\log_u_net_block_train_loss.txt'
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

plt.plot(list(range(0, len(list_loss_unet))), list_loss_unet,'blue', label='U-Net')
plt.plot(list(range(0, len(list_loss_unet_sa))), list_loss_unet_sa, 'r', label = 'ICSA-UNet')
# plt.xlabel('Epoch', weight='bold')
# plt.ylabel('Loss', weight='bold')
plt.xlabel('Epoch',)
plt.ylabel('Loss', )
plt.title('Validate Loss', fontsize=12, fontweight='bold')
plt.legend()
plt.show()