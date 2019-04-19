#!/usr/bin/env python
# -*- coding:utf-8 -*-

from __future__ import absolute_import
from __future__ import print_function

import torch
import torch.nn as nn


class RegLoss(nn.Module):
    def __init__(self):
        super(RegLoss, self).__init__()

    def forward(self, input, target):
        mask = torch.gt(target, 0).float()
        n = torch.sum(mask)
        input = torch.mul(input, mask)
        loss = torch.sum(torch.abs(input - target)) / n
        return loss


class Validation(nn.Module):
    def __init__(self):
        super(Validation, self).__init__()

    def forward(self, input, target):
        mask = torch.gt(target, 0).float()
        n = torch.sum(mask)
        input = torch.mul(input, mask)
        abs_loss = torch.abs(input - target)
        error_gt_2 = torch.sum(torch.gt(abs_loss, 2).float()) / n
        error_gt_3 = torch.sum(torch.gt(abs_loss, 3).float()) / n
        error_gt_5 = torch.sum(torch.gt(abs_loss, 5).float()) / n
        return error_gt_2, error_gt_3, error_gt_5


class Img_warp_loss(nn.Module):
    def __init__(self):
        super(Img_warp_loss.self).__init__()

    def forward(self, disp, left_img, right_img):
        B, C, H, W = disp.size()

        # mesh grid
        grid = torch.arange(0, W).view(1, -1).repeat(H, 1)
        grid = grid.view(1, 1, H, W).repeat(B, 1, 1, 1)

        vgrid = grid - disp

        vgrid = vgrid.permute(0, 2, 3, 1)
        output = nn.functional.grid_sample(right_img, vgrid)
        mask = torch.ones(right_img.size())
        mask = nn.functional.grid_sample(mask, vgrid)

        mask[mask < 0.999] = 0
        mask[mask > 0] = 1

        img_warp_mask = output * mask
        img_mask = left_img * mask
        img_mask.detach_()
        criterion = torch.nn.L1Loss()
        loss = criterion(img_warp_mask, img_mask)

        return loss, img_warp_mask
