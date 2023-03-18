import os.path
import shutil

import torch
import torch.nn as nn

from tensorboardX import SummaryWriter


class SE(nn.Module):
    def __init__(self, channel, ratio=4):
        super(SE, self).__init__()
        # 第一步：全局平均池化,输入维度(1,1,channel)
        self.avg = nn.AdaptiveAvgPool2d(1)
        # 第二步：全连接，通道数量缩减
        self.fc1 = nn.Linear(channel, channel // ratio, False)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(channel // ratio, channel, False)
        self.act2 = nn.Sigmoid()

    def forward(self, x):
        b, c, h, w = x.size()
        # b, c, 1, 1->b,c
        y = self.avg(x).view(b, c)
        y = self.fc1(y)
        y = self.act(y)
        y = self.fc2(y)
        y = self.act2(y).view(b, c, 1, 1)
        # print(y)
        return x * y


class SE_outline(nn.Module):
    def __init__(self, channel, ratio=4):
        super(SE_outline, self).__init__()
        # 第一步：全局平均池化,输入维度(1,1,channel)
        self.avg = nn.AdaptiveAvgPool2d(1)
        # 第二步：全连接，通道数量缩减
        self.fc1 = nn.Linear(channel, channel // ratio, False)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(channel // ratio, channel, False)
        self.act2 = nn.Sigmoid()

    def forward(self, x):
        b, c, h, w = x.size()
        # b, c, 1, 1->b,c
        y = self.avg(x).view(b, c)
        y = self.fc1(y)
        y = self.act(y)
        y = self.fc2(y)
        y = self.act2(y).view(b, c, 1, 1)
        # print(y)
        return y


class Attention_block(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(Attention_block, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)

        return x * psi


class conv_block(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(conv_block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class up_conv(nn.Module):
    # 采用的是上采样
    def __init__(self, ch_in, ch_out):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', ),
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x


class UNet(nn.Module):  # UNet AttU_Net
    def __init__(self, img_ch=7, output_ch=1):
        super(UNet, self).__init__()

        self.Maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)  #############
        self.Maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)  #############
        self.Maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)  #############
        self.Maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)  #############

        self.Maxpool5 = nn.MaxPool2d(kernel_size=2, stride=2)  #############
        self.Maxpool6 = nn.MaxPool2d(kernel_size=4, stride=4)  #############
        self.Maxpool7 = nn.MaxPool2d(kernel_size=8, stride=8)  #############
        # self.Maxpool8 = nn.MaxPool2d(kernel_size=2, stride=2)  #############

        self.AveragePool5 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.AveragePool6 = nn.AvgPool2d(kernel_size=4, stride=4)
        self.AveragePool7 = nn.AvgPool2d(kernel_size=8, stride=8)

        self.Con1_4 = nn.Conv2d(in_channels=14, out_channels=64, kernel_size=1)
        self.Con1_5 = nn.Conv2d(in_channels=14, out_channels=128, kernel_size=1)
        self.Con1_6 = nn.Conv2d(in_channels=14, out_channels=256, kernel_size=1)
        self.Con1_7 = nn.Conv2d(in_channels=14, out_channels=512, kernel_size=1)

        self.Conv1 = conv_block(ch_in=img_ch, ch_out=64)
        self.Conv2 = conv_block(ch_in=64, ch_out=128)
        self.Conv3 = conv_block(ch_in=128, ch_out=256)
        self.Conv4 = conv_block(ch_in=256, ch_out=512)
        self.Conv5 = conv_block(ch_in=512, ch_out=1024)

        self.Up5 = up_conv(ch_in=1024, ch_out=512)
        self.Att5 = Attention_block(F_g=512, F_l=512, F_int=256)
        self.Up_conv5 = conv_block(ch_in=1024, ch_out=512)

        self.Up4 = up_conv(ch_in=512, ch_out=256)
        self.Att4 = Attention_block(F_g=256, F_l=256, F_int=128)
        self.Up_conv4 = conv_block(ch_in=512, ch_out=256)

        self.Up3 = up_conv(ch_in=256, ch_out=128)
        self.Att3 = Attention_block(F_g=128, F_l=128, F_int=64)
        self.Up_conv3 = conv_block(ch_in=256, ch_out=128)

        self.Up2 = up_conv(ch_in=128, ch_out=64)
        self.Att2 = Attention_block(F_g=64, F_l=64, F_int=32)
        self.Up_conv2 = conv_block(ch_in=128, ch_out=64)

        self.Conv_1x1 = nn.Conv2d(64, output_ch, kernel_size=1, stride=1, padding=0)

        self.Th = torch.nn.Sigmoid()

        self.SE4 = SE_outline(channel=512)
        self.SE3 = SE_outline(channel=256)
        self.SE2 = SE_outline(channel=128)
        self.SE1 = SE_outline(channel=64)

    def forward(self, x):
        # encoding path
        x1 = self.Conv1(x)

        x2 = self.Maxpool1(x1)
        x2 = self.Conv2(x2)

        x3 = self.Maxpool2(x2)
        x3 = self.Conv3(x3)

        x4 = self.Maxpool3(x3)
        x4 = self.Conv4(x4)

        x5 = self.Maxpool4(x4)
        x5 = self.Conv5(x5)

        # decoding + concat path
        # 将g 改为图片的第一层

        ####################
        # size_feature_map = x.shape[-1]
        # outline = x[:, 1, :, :]
        # single_channel = outline.view(-1, 1, size_feature_map, size_feature_map)

        # 直接让其看
        # d2_outline_max = self.Maxpool5(x)
        # d2_outline_avg = self.AveragePool5(x)
        # d2_outline = torch.cat((d2_outline_max, d2_outline_avg), dim=1)

        d2_outline = torch.cat((x, x), dim=1)
        d2_outline = self.Con1_4(d2_outline)

        d3_outline_max = self.Maxpool5(x)
        d3_outline_avg = self.AveragePool5(x)
        d3_outline = torch.cat((d3_outline_max, d3_outline_avg), dim=1)
        d3_outline = self.Con1_5(d3_outline)

        d4_outline_max = self.Maxpool6(x)
        d4_outline_avg = self.AveragePool6(x)
        d4_outline = torch.cat((d4_outline_max, d4_outline_avg), dim=1)
        d4_outline = self.Con1_6(d4_outline)

        d5_outline_max = self.Maxpool7(x)
        d5_outline_avg = self.AveragePool7(x)
        d5_outline = torch.cat((d5_outline_max, d5_outline_avg), dim=1)
        d5_outline = self.Con1_7(d5_outline)

        # d2_outline = single_channel.repeat(1, 64, 1, 1)  # 64

        # d3_outline = single_channel.repeat(1, 128, 1, 1)
        # d3_outline = self.Maxpool5(d3_outline)
        #
        # d4_outline = single_channel.repeat(1, 256, 1, 1)
        # d4_outline = self.Maxpool6(d4_outline)
        #
        # d5_outline = single_channel.repeat(1, 512, 1, 1)
        # d5_outline = self.Maxpool7(d5_outline)

        # x4 = self.SE4(x4)
        y4 = self.SE4(d5_outline)
        x4_att = y4 * x4
        x4 = x4_att
        d5 = self.Up5(x5)
        # x4 = self.Att5(g=d5, x=x4)
        d5 = torch.cat((x4, d5), dim=1)
        d5 = self.Up_conv5(d5)

        y3 = self.SE3(d4_outline)
        x3_att = y3 * x3
        x3 = x3_att
        d4 = self.Up4(d5)
        # x3 = self.Att4(g=d4, x=x3)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        y2 = self.SE2(d3_outline)
        x2_att = y2 * x2
        x2 = x2_att
        d3 = self.Up3(d4)
        # x2 = self.Att3(g=d3, x=x2)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_conv3(d3)


        y1 = self.SE1(d2_outline)
        x1_att = y1 * x1
        x1 = x1_att
        d2 = self.Up2(d3)
        # x1 = self.Att2(g=d2, x=x1)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_conv2(d2)

        d1 = self.Conv_1x1(d2)

        return self.Th(d1)

        # return d1


if __name__ == '__main__':

    model = UNet()
    input = torch.rand(2, 7, 512, 512)
    y = model(input)
    print(y.shape)

    # k = input[:, 1, :, :].reshape(-1, 1, 512, 512)
    # print(k.shape)

    path_netVisual_file = '../network_visualization'
    if os.path.exists(path_netVisual_file):
        shutil.rmtree(path_netVisual_file)
        os.makedirs(path_netVisual_file)

    with SummaryWriter(logdir="network_visualization") as w:
        w.add_graph(model, input)
