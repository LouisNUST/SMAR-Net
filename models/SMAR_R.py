# -*- coding: utf-8 -*-
from __future__ import division, absolute_import, print_function

import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def conv2d(in_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=True):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
        nn.BatchNorm2d(out_planes),
        nn.LeakyReLU()
    )


def conv3d(in_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=True):
    return nn.Sequential(
        nn.Conv3d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
        nn.BatchNorm3d(out_planes),
        nn.LeakyReLU()
    )


def convtrans2d(in_planes, out_planes, kernel_size=3, stride=1, padding=0, output_padding=0, bias=True):
    return nn.Sequential(
        nn.ConvTranspose2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,
                           output_padding=output_padding, bias=bias),
        nn.BatchNorm2d(out_planes),
        nn.LeakyReLU()
    )


def convtrans3d(in_planes, out_planes, kernel_size=3, stride=1, padding=0, output_padding=0, bias=True):
    return nn.Sequential(
        nn.ConvTranspose3d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,
                           output_padding=output_padding, bias=bias),
        nn.BatchNorm3d(out_planes),
        nn.LeakyReLU()
    )


def deconv3d(in_planes, out_planes, kernel_size=3, stride=1, padding=1, output_padding=1, bias=True):
    return nn.Sequential(
        nn.Conv3d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
        nn.BatchNorm3d(out_planes),
        nn.LeakyReLU(),
        nn.Conv3d(out_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
        nn.BatchNorm3d(out_planes),
        nn.LeakyReLU(),
        nn.ConvTranspose3d(out_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,
                           output_padding=output_padding, bias=bias),
        nn.BatchNorm3d(out_planes),
        nn.LeakyReLU()
    )


def predict_disp(in_planes):
    return nn.Sequential(
        nn.Conv3d(in_planes, 32, kernel_size=3, stride=1, padding=1, bias=True),
        nn.BatchNorm3d(32),
        nn.LeakyReLU(),
        nn.Conv3d(32, 16, kernel_size=3, stride=1, padding=1, bias=True),
        nn.BatchNorm3d(16),
        nn.LeakyReLU(),
        nn.ConvTranspose3d(16, 1, kernel_size=3, stride=2, padding=1, output_padding=1)
    )


class SMAR_R(nn.Module):
    def __init__(self, disp_max=128):
        super(SMAR_R, self).__init__()
        self.disp_max = disp_max

        self.conv0_1 = conv3d(in_planes=6, out_planes=32, kernel_size=5, stride=2, padding=2)
        # self.conv0_2 = conv3d(in_planes=32, out_planes=32, kernel_size=3, stride=2, padding=1)
        # self.conv0_3 = conv3d(in_planes=32, out_planes=32, kernel_size=3, stride=1, padding=1)

        self.conv1_1 = conv3d(in_planes=32, out_planes=32, kernel_size=3, stride=1, padding=1)
        self.conv1_2 = conv3d(in_planes=32, out_planes=32, kernel_size=3, stride=1, padding=1)
        self.conv1_a = conv3d(in_planes=32, out_planes=32, kernel_size=1, stride=2, padding=0)
        self.conv1_b = conv3d(in_planes=32, out_planes=32, kernel_size=3, stride=2, padding=1)
        self.conv1_c = conv3d(in_planes=32, out_planes=32, kernel_size=5, stride=2, padding=2)
        self.conv1 = conv3d(32 * 3, 32, 3, 1, 1)

        self.conv2_1 = conv3d(in_planes=32, out_planes=64, kernel_size=3, stride=1, padding=1)
        self.conv2_2 = conv2d(in_planes=64, out_planes=64, kernel_size=3, stride=1, padding=1)
        self.conv2_a = conv3d(in_planes=64, out_planes=64, kernel_size=1, stride=2, padding=0)
        self.conv2_b = conv3d(in_planes=64, out_planes=64, kernel_size=3, stride=2, padding=1)
        self.conv2_c = conv3d(in_planes=64, out_planes=64, kernel_size=5, stride=2, padding=2)
        self.conv2 = conv3d(64 * 3, 64, 3, 1, 1)

        self.conv3_1 = conv3d(in_planes=64, out_planes=64, kernel_size=3, stride=1, padding=1)
        self.conv3_2 = conv2d(in_planes=64, out_planes=64, kernel_size=3, stride=1, padding=1)
        self.conv3_a = conv3d(in_planes=64, out_planes=64, kernel_size=1, stride=2, padding=0)
        self.conv3_b = conv3d(in_planes=64, out_planes=64, kernel_size=3, stride=2, padding=1)
        self.conv3_c = conv3d(in_planes=64, out_planes=64, kernel_size=5, stride=2, padding=2)
        self.conv3 = conv3d(64 * 3, 64, 3, 1, 1)

        self.conv4_1 = conv3d(in_planes=64, out_planes=128, kernel_size=3, stride=1, padding=1)
        self.conv4_2 = conv2d(in_planes=128, out_planes=128, kernel_size=3, stride=1, padding=1)
        self.conv4_a = conv3d(in_planes=128, out_planes=128, kernel_size=1, stride=2, padding=0)
        self.conv4_b = conv3d(in_planes=128, out_planes=128, kernel_size=3, stride=2, padding=1)
        self.conv4_c = conv3d(in_planes=128, out_planes=128, kernel_size=5, stride=2, padding=2)
        self.conv4 = conv3d(128 * 3, 128, 3, 1, 1)

        self.deconv4 = deconv3d(in_planes=128, out_planes=128)
        self.deconv3 = deconv3d(in_planes=128 * 2, out_planes=64)
        self.deconv2 = deconv3d(in_planes=64 * 2, out_planes=64)
        self.deconv1 = deconv3d(in_planes=64 * 2, out_planes=32)
        self.predict_0 = predict_disp(in_planes= 32)
        self.predict_1 = predict_disp(in_planes= 64)
        self.predict_2 = predict_disp(in_planes= 64)

        for m in self.modules():
            if isinstance(m,nn.Conv3d) or isinstance(m, nn.ConvTranspose3d):
                nn.init.kaiming_normal_(m.weight.data, mode = 'fan_in')
                if m.bias is not None:
                    m.bias.data_zero_()

    def forward(self, left_img, right_img):
        out = self.concat(left_img, right_img)
        out = self.conv0_1(out)
        # out = self.conv0_2(out)
        # out = self.conv0_3(out)

        c11 = self.conv1_2(self.conv1_1(out))
        c12 = self.conv1(torch.cat((self.conv1_a(c11), self.conv1_b(c11), self.conv1_c(c11)), dim=1))

        c21 = self.conv2_2(self.conv2_1(c12))
        c22 = self.conv2(torch.cat((self.conv2_a(c21), self.conv2_b(c21), self.conv2_c(c21)), dim=1))

        c31 = self.conv3_1(self.conv3_2(c22))
        c32 = self.conv3(torch.cat((self.conv3_a(c31), self.conv3_b(c31), self.conv3_c(c31)), dim=1))

        c41 = self.conv4_2(self.conv4_1(c32))
        c42 = self.conv4(torch.cat((self.conv4_a(c41), self.conv4_b(c41), self.conv4_c(c41)), dim=1))

        d42 = self.deconv4(c42)
        d41 = torch.cat((d42,c41),dim=1)

        d32 = self.deconv3(d41)
        d31 = torch.cat((d32,c31),dim = 1)

        d22 = self.deconv2(d31)
        d21 = torch.cat((d22,c21),dim = 1)

        d12 = self.deconv1(d21)
        d11 = torch.cat((d12,c11),dim = 1)

        cost_0 = self.predict_0(d11)
        cost_1 = self.predict_1(d21)
        cost_2 = self.predict_2(d31)

        disp_0 = self.soft_argmin(cost_0)
        disp_1 = self.soft_argmin(cost_1)
        disp_2 = self.soft_argmin(cost_2)

        if self.training:
            return disp_2,disp_1,disp_0
        else:
            return disp_0


    def concat(self, left_img, right_img):
        B, C, H, W = left_img.size()
        concat = torch.zeros((B, 2 * C, self.disp_max, H, W))
        for d in range(self.disp_max):
            concat[:, :C, d, :, :] = left_img
            concat[:, C:, d, :, d:W] = right_img[:, :, :, :W - d]
        return concat

    def soft_argmin(self, cost_volume):
        pro_volume = -cost_volume
        pro_volume = torch.squeeze(pro_volume, dim=0)
        pro_volume = torch.squeeze(pro_volume, dim=0)

        pro_volume = F.softmax(pro_volume)
        d = pro_volume.size()[0]
        pro_volume_indices = torch.zeros(pro_volume.size())

        for i in range(d):
            pro_volume_indices[i] = i * pro_volume[i]
            disp_prediction = torch.sum(pro_volume_indices, dim=0)
            disp_prediction = disp_prediction.unsqueeze(dim=0)
            disp_prediction = disp_prediction.unsqueeze(dim=0)
        return disp_prediction
