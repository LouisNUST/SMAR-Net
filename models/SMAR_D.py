# -*- coding:utf-8 -*-

from __future__ import print_function, absolute_import, division

import torch
import torch.nn as nn


def conv2d(in_planes, out_planes, kernel_size=3, stride=2, padding=1):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding),
        nn.BatchNorm2d(out_planes),
        nn.LeakyReLU(0.2)
    )


class SMAR_D(nn.Module):
    def __init__(self):
        super(SMAR_D, self).__init__()

        self.conv1 = conv2d(in_planes=1, out_planes=32, kernel_size=3, stride=2, padding=1)

        self.conv2 = conv2d(in_planes=32, out_planes=32, kernel_size=3, stride=2, padding=1)
        self.conv2a = conv2d(in_planes=32, out_planes=4, kernel_size=1, stride=1, padding=0)

        self.conv3 = conv2d(in_planes=32, out_planes=64, kernel_size=3, stride=2, padding=1)
        self.conv3a = conv2d(in_planes=64, out_planes= 8, kernel_size=1, stride=1, padding=0)

        self.conv4 = conv2d(in_planes=64, out_planes=128, kernel_size=3, stride=2, padding=1)
        self.conv4a = conv2d(in_planes=128,out_planes= 8, kernel_size=1,stride=1, padding=1)

        self.conv5 = conv2d(in_planes=128, out_planes=128, kernel_size=3, stride=2, padding=1)
        self.conv5a = conv2d(in_planes=128, out_planes=8, kernel_size=1, stride =1,padding = 1 )


        self.fc1 = nn.Sequential(
            nn.Linear(6784, 1024),
            nn.ReLU()
        )

        self.fc2 = nn.Sequential(
            nn.Linear(1024, 256),
            nn.ReLU()
        )

        self.fc3 = nn.Sequential(
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        out_1 = self.conv1(x)
        out_2 = self.conv2(out_1)
        out_2a = self.conv2(out_2)
        out_2b = out_2a.view(-1,self.flat(out_2a))
        out_3 = self.conv3(out_2)
        out_3a = self.conv2(out_3)
        out_3b = out_3a.view(-1,self.flat(out_3a))
        out_4 = self.conv2(out_3)
        out_4a = self.conv2(out_4)
        out_4b = out_4a.view(-1,self.flat(out_4a))
        out_5 = self.conv5(out_4)
        out_5a = self.conv2(out_5)
        out_5b = out_5a.view(-1,self.flat(out_5a))

        out = torch.cat(out_2b,out_3b,out_4b,out_5b)
        out = self.fc1(out)
        out = self.fc2(out)
        out = self.fc3(out)

        return out

    def flat(self,x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


