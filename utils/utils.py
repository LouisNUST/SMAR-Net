# -*- coding:utf-8 -*-
from __future__ import print_function, division, absolute_import

import datetime

from path import Path


def save_path_formatter(args):
    args_dict = vars(args)
    dataset_name = str(Path(args_dict['dataset_name']))
    folder_string = [dataset_name]
    folder_string.append(str(Path(args_dict['model_name'])))
    save_path = Path('-'.join(folder_string))
    timestamp = datetime.datetime.now().strftime("%m-%d-%H:%M")
    if args.is_debug:
        return save_path / 'debug'
    else:
        return save_path / timestamp


class AverageMeter(object):

    def __init__(self, i=1):
        self.meters = i
        self.reset(self.meters)

    def reset(self, i):
        self.avg = [0] * i
        self.sum = [0] * i
        self.count = 0

    def update(self, val, n=1):
        # n: batch_size; v: average value of each batch
        if not isinstance(val, list):
            val = [val]
        assert (len(val) == self.meters)
        self.count += n
        for i, v in enumerate(val):
            self.sum[i] += v * n
            self.avg[i] = self.sum[i] / self.count
