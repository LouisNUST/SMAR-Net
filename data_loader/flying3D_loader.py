# -*- coding:utf-8 -*-

from __future__ import print_function, division, absolute_import

import os

import numpy as np
from PIL import Image
from path import Path
from torch.utils.data import Dataset


class SMARDataLoader(Dataset):
    def __init__(self, transform=None, train=True):
        super(SMARDataLoader, self).__init__()
        self.dataset_root = Path('/data/SMAR-Dataset/SceneFlow/Flying3D/')
        self.train = train
        self.get_filelist()
        self.transform = transform

    def get_filelist(self):
        self.left_img_list = []
        self.right_img_list = []
        self.img_root = self.dataset_root + 'frames_cleanpass/'

        if self.train:
            mode = 'train/'
        else:
            mode = 'valid/'

        img_dir = self.img_root + mode
        subdir = ['A/', 'B/', 'C/']
        for ss in subdir:
            imgs = os.listdir(img_dir + ss)
            for dd in imgs:
                img_l = os.listdir(img_dir + ss + dd + '/left/')
                for img_name in img_l:
                    self.left_img_list.append(img_dir + ss + dd + '/left/' + img_name)
                    self.right_img_list.append(img_dir + ss + dd + '/right/' + img_name)

    def __getitem__(self, index):

        left_img = Image.open(self.left_img_list[index])
        right_img = Image.open(self.right_img_list[index])
        left_img, right_img = np.asarray(left_img), np.asarray(right_img)

        out_left_img, out_right_img = self.transform(left_img, right_img)

        return out_left_img, out_right_img

    def __len__(self):
        return len(self.left_img_list)
