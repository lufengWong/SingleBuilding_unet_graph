"""
训练器模块
"""
import os
import shutil
import time

# import unet_building as unet
from network_candidates import unet_building as unet
import torch
from data import WallDataset
import torch.nn as nn

from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision.utils import save_image

import numpy as np

from function_evaluate import SegmentationMetric


def dice_coeff(pred, target):
    smooth = 1.
    # num = pred.size(1)
    num_shape = pred.shape[1]
    m1 = pred.reshape(num_shape, -1)  # Flatten
    m2 = target.reshape(num_shape, -1)  # Flatten
    intersection = (m1 * m2).sum()

    # 损失
    return (2. * intersection + smooth) / (m1.sum() + m2.sum() + smooth)


def log(file, msg='', is_print=True):
    if is_print:
        print(msg)
    file.write(msg + '\n')
    file.flush()
    ##############


def val(model, dataloader, file):
    model.eval()
    predict_accuracy = 0
    num_predict_wall_right = 0
    num_target_wall = 0
    num_predict_wall = 0

    num_dice = 0

    pa = 0
    cpa = 0
    mpa = 0
    mIoU = 0

    for _, (input, target) in enumerate(dataloader):
        batch_size = input.shape[0]
        with torch.no_grad():  # 在该模块下，所有计算得出的tensor的requires_grad都自动设置为False。
            input = input.cuda()
            output = model(input).cuda()

        output = output.to(torch.float32)
        target = target.to(torch.float32).cuda()

        loss_func = nn.BCELoss().to('cuda')
        loss_bce = loss_func(output, target)

        log(file, f'loss_train: {loss_bce}')

        output = output.cpu().numpy()
        target = target.cpu().numpy()
        # predict = np.argmax(output, axis=1)
        predict = output

        for i in range(batch_size):
            # 累加所有
            # 评价指标
            predict_this = predict[i]
            target_this = target[i]

            num_dice += dice_coeff(predict_this, target_this)

            predict_this[predict_this < 0.5] = 0
            predict_this[predict_this >= 0.5] = 1

            predict_int = predict_this.astype('int')
            target_int = target_this.astype('int')

            # print(predict_int)
            # print(target_int)

            metric = SegmentationMetric(2)
            metric.addBatch(predict_int, target_int)
            pa += metric.pixelAccuracy()
            # cpa += metric.classPixelAccuracy()
            mpa += metric.meanPixelAccuracy()  ######################
            mIoU += metric.meanIntersectionOverUnion()

            num_predict = np.sum(predict_int == target_int)  # 像素点对的个数
            predict_accuracy += (num_predict / (input.shape[2] * input.shape[3]))  # 累计的预测对的百分比
            num_predict_wall_right += np.sum((predict_int == 0) & (target_int == 0))  # 为0，累计墙对的个数
            num_target_wall += np.sum(target_int == 0)  # 目标墙的个数
            num_predict_wall += np.sum(predict_int == 0)  # 生成墙的个数

    model.train()

    predict_accuracy = round(predict_accuracy / len(dataloader.dataset), 5)
    num_predict_wall_right = int(num_predict_wall_right / len(dataloader.dataset))  #
    num_target_wall = int(num_target_wall / len(dataloader.dataset))
    num_predict_wall = int(num_predict_wall / len(dataloader.dataset))

    num_datas = len(dataloader.dataset)
    num_dice = round(num_dice / num_datas, 5)
    num_pa = round(pa / num_datas, 5)
    # num_cpa = round(cpa/num_datas, 5)
    num_mpa = round(mpa / num_datas, 5)
    num_mIou = round(mIoU / num_datas, 5)

    if num_target_wall != 0:
        wall_accuracy = round(num_predict_wall_right / num_target_wall, 5)  # 墙预测目标的准确率
    else:
        wall_accuracy = "nan"
    if num_predict_wall != 0:
        wall_proportion = round(num_predict_wall_right / num_predict_wall, 5)  # 墙预测中自己对的个数
    else:
        wall_proportion = "nan"

    # 平均
    log(file, f'Predict Accuracy: {predict_accuracy}')
    log(file, f'Number of Predict Wall Right: {num_predict_wall_right}')
    log(file, f'Number of Target Wall: {num_target_wall}')
    log(file, f'Number of Predict Wall: {num_predict_wall}')
    log(file, f'Wall Accuracy(right-wall/target-wall)-important: {wall_accuracy}')
    log(file, f'Wall Proportion(right-wall/predict-wall): {wall_proportion}')

    log(file, f'Dice: {num_dice}')
    log(file, f'pa: {num_pa}')
    # log(file, f'cpa: {num_cpa}')
    log(file, f'mpa: {num_mpa}')
    log(file, f'mIoU: {num_mIou}')
    log(file, f'loss_val: {loss_bce}')

    return predict_accuracy, num_predict_wall_right, num_target_wall, num_predict_wall, wall_accuracy, wall_proportion, loss_bce


# 训练器
class Trainer:

    def __init__(self, path_train, path_val, model, model_copy, img_save_path, log_file):
        self.path_train = path_train
        self.path_val = path_val
        self.model = model
        self.model_copy = model_copy
        self.img_save_path = img_save_path
        self.log_file = log_file
        # 使用的设备
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # self.device = torch.device ('cpu')
        # 网络
        self.net = unet.UNet().to(self.device)
        # 优化器，这里用的Adam，跑得快点
        self.opt = torch.optim.Adam(self.net.parameters())
        # 这里直接使用二分类交叉熵来训练，效果可能不那么好
        # nn.BCELoss()
        # 可以使用其他损失，比如DiceLoss、FocalLoss之类的
        self.loss_func = nn.BCELoss()  # 二分类问题
        # 设备好，batch_size和num_workers可以给大点
        self.loader = DataLoader(WallDataset(data_root=path_train, mask_size=2), batch_size=8, shuffle=True,
                                 num_workers=4)
        # add
        self.loader_val = DataLoader(WallDataset(data_root=path_val, mask_size=2), batch_size=8, shuffle=True,
                                     num_workers=4)

        if os.path.exists(self.img_save_path):
            shutil.rmtree(self.img_save_path)
            os.mkdir(self.img_save_path)
        else:
            os.mkdir(self.img_save_path)
        # # 判断是否存在模型
        # if os.path.exists(self.model):
        #     self.net.load_state_dict(torch.load(model))
        #     print(f"Loaded{model}!")
        # else:
        #     print("No Param!")
        # os.makedirs(img_save_path, exist_ok=True)

    # 训练
    def train(self, stop_value):
        t = time.localtime()

        list_loss = []
        log_file = open(self.log_file, 'w')
        log(log_file, f'------------------------------ ')
        log(log_file, str((t.tm_year, t.tm_mon, t.tm_mday, t.tm_hour, t.tm_min, t.tm_sec)))
        log(log_file, str(self.model))
        log(log_file, f'------------------------------ ')

        epoch = 1
        overFit = False
        while True:

            if overFit:
                break

            for inputs, labels in tqdm(self.loader, desc=f"Epoch {epoch}/{stop_value}", ascii=True,
                                       total=len(self.loader)):  # 同时有两种 #################################
                # 图片和分割标签
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                # 输出生成的图像
                out = self.net(inputs)

                out = out.to(torch.float32)
                labels = labels.to(torch.float32)

                loss_bce = self.loss_func(out, labels)
                # loss_dice = dice_coeff(out, labels)

                loss = loss_bce

                # 后向
                self.opt.zero_grad()
                loss.backward()
                self.opt.step()

                # 输入的图像，取第一张 ####每一次batch取第一张
                x = (inputs[0][0, :, :] + inputs[0][2, :, :]).view(1, 256, 256)
                # 生成的图像，取第一张
                x_ = out[0]
                # 标签的图像，取第一张
                y = labels[0]
                # 三张图，从第0轴拼接起来，再保存
                img = torch.stack([x, x_, y], 0)
                save_image(img.cpu(), os.path.join(self.img_save_path, f"{epoch}.png"))  # 存下了此batch的最后一张
                # print("image save successfully !")
            print(f"\nEpoch: {epoch}/{stop_value}, Train Loss: {loss}")
            # torch.save(self.net.state_dict(), self.model)
            # print("model is saved !")

            if epoch % 1 == 0:

                log(log_file, f' ')
                log(log_file, f'------------------------------ ')
                log(log_file, f'Epoch: {epoch}')

                loss_val = val(self.net, self.loader_val, log_file)[6]
                list_loss.append(loss_val)

                num_tolerance = 15  # 多少代没有变化就停止，防止过拟合
                if len(list_loss) >= num_tolerance + 2:
                    min_now = min(list_loss[-num_tolerance:])
                    min_before = min(list_loss[:-num_tolerance])
                    print('loss_val >>> min_before: % s ; min_now: %s' % (min_before, min_now))

                    if min_now > min_before:
                        torch.save(self.net.state_dict(), self.model_copy.format(epoch, loss))
                        print("model_copy is saved !")
                        overFit = True
            # 备份
            if epoch % 15 == 0:
                torch.save(self.net.state_dict(), self.model_copy.format(epoch, loss))
                # torch.save(self.net, 'model_all.h5')  # 保存整个网络
                print("model_copy is saved !")

            if epoch > stop_value:
                torch.save(self.net.state_dict(), self.model_copy.format(epoch, loss))
                # torch.save(self.net, 'model_all.h5')  # 保存整个网络
                print("model_copy is saved !")
                break
            epoch += 1


if __name__ == '__main__':

    name = 'u_net_baseline_train_loss'  # ##########

    path_train_files = r"F:\dataset_U-net\train_reinforce"
    path_val_files = r'F:\dataset_U-net\val'
    path_net_save = r'F:\U-net-train-val-test\model_trained'
    path_log_save = r'C:\Users\Administrator\Desktop\singleBuilding_unet_graph\Part_2_SingleBuilding_Unet\log'

    path_img_save = os.path.join(r'F:\U-net-train-val-test\val_img_generated', name)

    if os.path.exists(path_img_save):
        shutil.rmtree(path_img_save)
    os.makedirs(path_img_save)

    t = Trainer(path_train=path_train_files,
                path_val=path_val_files,
                model=os.path.join(path_net_save, 'model_' + name + '.pth'),
                model_copy=os.path.join(path_net_save, 'model_{}_{}_UNet_' + name + '.pth'),
                img_save_path=path_img_save,
                log_file=os.path.join(path_log_save, 'log_' + name + '.txt'))

    t.train(200)
