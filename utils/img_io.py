# -*- coding:utf-8 -*-

from __future__ import absolute_import
from __future__ import print_function
import os
import sys
import re
import numpy as np
import torch
from PIL import Image


def load_pfm(file_name, downsample= False):
    if downsample:
        pass

    file = open(file_name)

    color = None
    width = None
    height = None
    scale = None
    endian = None

    header = file.readline().rstrip()

    if header == 'PF':
        color = True
    elif header == 'Pf':
        color = False
    else:
        raise Exception('Not a PFM file.')

    dim_match = re.match(r'^(\d+)\s(\d+)\s$', file.readline())
    if dim_match:
        width, height = map(int, dim_match.groups())
    else:
        raise Exception('Malformed PFM header.')

    scale = float(file.readline().rstrip())
    if scale < 0:
        endian = '<'
        scale = -scale
    # big-endian
    else:
        endian = '>'

    data = np.fromfile(file, endian + 'f')
    shape = (height, width, 3) if color else (height, width)
    data = np.reshape(data,shape)
    img = np.flipud(data)
    file.close()

    return img, scale


def save_pfm(file_name, image, scale=1):
    file = open(file_name, 'w')
    color = None

    if image.dtype.name != 'float32':
        raise Exception('Image dtype must be float32.')

    if len(image.shape) == 3 and image.shape[2] == 3:
        color = True
    elif len(image.shape) == 2 or len(image.shape) == 3 and image.shape[2] == 1:
        color = False
    else:
        raise Exception('Image must have H x W x 3, H x W x 1 or H x W dimensions.')

    file.write('PF\n' if color else 'Pf\n')
    file.write('%d %d\n' % (image.shape[1], image.shape[0]))

    endian = image.dtype.byteorder

    if endian == '<' or endian == '=' and sys.byteorder == 'little':
        scale = -scale

    file.write('%f\n' % scale)

    np.flipud(image).tofile(file)


def disp_map(disp):
    map = np.array([
        [0,0,0,114],
        [0,0,1,185],
        [1,0,0,114],
        [1,0,1,174],
        [0,1,0,114],
        [0,1,1,185],
        [1,1,0,114],
        [1,1,1,0]
    ])
    bins = map[0:map.shape[0]-1,map.shape[1] - 1].astype(float)
    bins = bins.reshape((bins.shape[0], 1))
    cbins = np.cumsum(bins)
    bins = bins / cbins[cbins.shape[0] -1]
    cbins = cbins[0:cbins.shape[0]-1] / cbins[cbins.shape[0] -1]
    cbins = cbins.reshape((cbins.shape[0], 1))
    ind = np.tile(disp.T,(6,1))
    tmp = np.tile(cbins,(1,disp.size))
    b = (ind > tmp).astype(int)
    s = np.sum(b, axis= 0)
    bins = 1 / bins
    t = cbins
    cbins = np.zeros((cbins.size+1,1))
    cbins[1:] = t
    disp = (disp - cbins[s]) * bins[s]
    disp = map[s,0:3] * np.tile(1 - disp,(1,3)) + map[s + 1,0:3] * np.tile(disp,(1,3))
    return disp

def disp_to_color(disp):
    pre_shape = disp.shape
    max_disp = np.max(disp)
    disp = disp / max_disp
    disp = disp.reshape((disp.size,1))
    disp = disp_map(disp)
    disp = disp.reshape((pre_shape[0],pre_shape[1],3))
    disp = disp * 255.0
    return disp

def error_c(disp_gt, disp_est):
    disp_gt = torch.from_numpy(disp_gt).float()
    disp_est = torch.from_numpy(disp_est).float()
    mask = torch.gt(disp_gt, 0).float()
    n = torch.sum(mask)
    disp_est = torch.mul(disp_est, mask)
    abs_loss = torch.abs(disp_est - disp_gt)
    loss = torch.sum(abs_loss)
    error_gt_2 = torch.sum(torch.gt(abs_loss, 2).float())
    error_gt_3 = torch.sum(torch.gt(abs_loss, 3).float())
    error_gt_5 = torch.sum(torch.gt(abs_loss, 5).float())
    return  error_gt_2, error_gt_3, error_gt_5, n, loss

def disp_err_image(disp_est, disp_gt):
    disp_shape = disp_gt.shape
    cols = np.array([
        [0/3.0,       0.1875,  49,  54, 149],
        [0.1875/3.0,  0.375,   69, 117, 180],
        [0.375/3.0,   0.75,   116, 173, 209],
        [0.75/3.0,    1.5,    171, 217, 233],
        [1.5/3.0,     3,      224, 243, 248],
        [3/3.0,       6,      254, 224, 144],
        [6/3.0,      12,      253, 174,  97],
        [12/3.0,      24,      244, 109,  67],
        [24/3.0,      48,      215,  48,  39],
        [48/3.0,  float("inf") ,      165,   0,  38]
    ])
    tau = np.array([3.0, 0.05])

    E =  np.abs(disp_est - disp_gt)
    t1 = E / tau[0]
    t2 = E
    for i in range(E.shape[0]):
        for j in range(E.shape[1]):
            if disp_gt[i,j] > 0:
                t2[i,j] = E[i,j] / disp_gt[i,j] / tau[1]
    E = np.minimum(t1, t2)

    disp_err = np.zeros((disp_shape[0], disp_shape[1], 3),dtype= 'uint8')
    for c_i in range(cols.shape[0]):
        for i in range(disp_shape[0]):
            for j in range(disp_shape[1]):
                if disp_gt[i,j] != 0 and E[i,j] >= cols[c_i, 0] and E[i,j] <= cols[c_i, 1]:
                    disp_err[i,j,0] = int(cols[c_i,2])
                    disp_err[i,j,1] = int(cols[c_i,3])
                    disp_err[i,j,2] = int(cols[c_i,4])
    return disp_err

