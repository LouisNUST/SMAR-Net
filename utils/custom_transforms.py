# -*- coding: utf-8 -*-
from __future__ import print_function, division
import random
import numpy as np
import torch


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, left_img, right_img, left_disp):
        for t in self.transforms:
            left_img, right_img, left_disp = t(left_img, right_img, left_disp)
        return left_img, right_img, left_disp


class RandomCrop(object):

    def __init__(self, height, width):
        self.height = height
        self.width = width

    def __call__(self, left_img, right_img):
        in_h, in_w, in_c = left_img.shape  # H,W,C

        h = np.random.randint(0, in_h - self.height)
        w = np.random.randint(0, in_w - self.width)
        right_img_crop = np.zeros((self.height, self.width + 128))
        right_img_crop[:, max(0, w - 128) - (w - 128):, :] = right_img[h:h + self.height,
                                                               max(0, w - 127):w + self.width, :]
        left_img_crop = left_img[ h:h+self.height, w:w+self.width,:],

        return left_img_crop,right_img_crop


class ArrayToTensor(object):

    def __call__(self, left_img, right_img):
        left_img = np.transpose(left_img, (2, 0, 1))
        right_img = np.transpose(right_img, (2, 0, 1))

        left_img = torch.from_numpy(left_img).float() / 255
        right_img = torch.from_numpy(right_img).float() / 255

        return left_img, right_img
