# -*- coding: utf-8 -*-

"""
@date: 2020/3/31 下午7:48
@file: smooth_l1_loss.py
@author: zj
@description: 
"""

import torch
import torch.nn as nn


class SmoothL1Loss(nn.Module):

    def __init__(self):
        super(SmoothL1Loss, self).__init__()

    def forward(self, preds, targets):
        """
        计算边界框回归损失。N表示RoI个数
        :param preds: 大小为[N, 4]，分别保存每个边界框的预测x,y,w,h
        :param targets: 大小为[N, 4]，分别保存每个边界框的实际x,y,w,h
        :return:
        """
        res = self.smooth_l1(preds - targets)
        return torch.sum(res)

    def smooth_l1(self, x):
        if torch.abs(x) < 1:
            return 0.5 * torch.pow(x, 2)
        else:
            return torch.abs(x) - 0.5
