import os.path
import shutil

import torch
import torch.nn as nn

from tensorboardX import SummaryWriter


class spacialAttention_block(nn.Module):
    def __init__(self, h_input, h_x, c_input):
        super(spacialAttention_block, self).__init__()
        size_scale = int(h_input / h_x)
        self.MaxPool = nn.MaxPool2d(kernel_size=size_scale, stride=size_scale)
        self.AveragePool = nn.AvgPool2d(kernel_size=size_scale, stride=size_scale)
        self.one_channel = nn.Conv2d(in_channels=c_input * 2, out_channels=1, kernel_size=1)
        self.sig = nn.Sigmoid()

    def forward(self, feature_input):
        max_feature = self.MaxPool(feature_input)
        avg_feature = self.AveragePool(feature_input)
        combine_feature = torch.cat((max_feature, avg_feature), dim=1)
        y = self.one_channel(combine_feature)
        y = self.sig(y)
        return y


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

        self.img_ch = img_ch
        self.output_ch = output_ch

        self.Maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)  #############
        self.Maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)  #############
        self.Maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)  #############
        self.Maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)  #############

        self.Conv1 = conv_block(ch_in=img_ch, ch_out=64)
        self.Conv2 = conv_block(ch_in=64, ch_out=128)
        self.Conv3 = conv_block(ch_in=128, ch_out=256)
        self.Conv4 = conv_block(ch_in=256, ch_out=512)
        self.Conv5 = conv_block(ch_in=512, ch_out=1024)

        self.Up5 = up_conv(ch_in=1024, ch_out=512)
        # self.Att5 = Attention_block(F_g=512, F_l=512, F_int=256)
        self.Up_conv5 = conv_block(ch_in=1024, ch_out=512)

        self.Up4 = up_conv(ch_in=512, ch_out=256)
        # self.Att4 = Attention_block(F_g=256, F_l=256, F_int=128)
        self.Up_conv4 = conv_block(ch_in=512, ch_out=256)

        self.Up3 = up_conv(ch_in=256, ch_out=128)
        # self.Att3 = Attention_block(F_g=128, F_l=128, F_int=64)
        self.Up_conv3 = conv_block(ch_in=256, ch_out=128)

        self.Up2 = up_conv(ch_in=128, ch_out=64)
        # self.Att2 = Attention_block(F_g=64, F_l=64, F_int=32)
        self.Up_conv2 = conv_block(ch_in=128, ch_out=64)

        self.Conv_1x1 = nn.Conv2d(64, output_ch, kernel_size=1, stride=1, padding=0)

        self.Th = torch.nn.Sigmoid()

        self.spatialAttention_4 = spacialAttention_block(h_input=512, h_x=64, c_input=self.img_ch)
        self.spatialAttention_3 = spacialAttention_block(h_input=512, h_x=128, c_input=self.img_ch)
        self.spatialAttention_2 = spacialAttention_block(h_input=512, h_x=256, c_input=self.img_ch)
        self.spatialAttention_1 = spacialAttention_block(h_input=512, h_x=512, c_input=self.img_ch)


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

        y4 = self.spatialAttention_4(x)
        x4_att = y4 * x4
        x4 = x4_att + x4
        d5 = self.Up5(x5)
        d5 = torch.cat((x4, d5), dim=1)
        d5 = self.Up_conv5(d5)

        y3 = self.spatialAttention_3(x)
        x3_att = y3 * x3
        x3 = x3_att + x3
        d4 = self.Up4(d5)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        y2 = self.spatialAttention_2(x)
        x2_att = y2 * x2
        x2 = x2_att + x2
        d3 = self.Up3(d4)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        y1 = self.spatialAttention_1(x)
        x1_att = y1 * x1
        x1 = x1_att + x1
        d2 = self.Up2(d3)
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

    path_netVisual_file = '../network_visualization'
    if os.path.exists(path_netVisual_file):
        shutil.rmtree(path_netVisual_file)
        os.makedirs(path_netVisual_file)

    with SummaryWriter(logdir="network_visualization") as w:
        w.add_graph(model, input)
