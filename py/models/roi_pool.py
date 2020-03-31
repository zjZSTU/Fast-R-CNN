# -*- coding: utf-8 -*-

"""
@date: 2020/3/31 下午3:17
@file: roi_pool.py
@author: zj
@description: 
"""

import torch.nn as nn


class ROI_Pool(nn.Module):

    def __init__(self, size):
        super(ROI_Pool, self).__init__()
        assert len(size) == 2, 'size参数输入(长, 宽)'
        pool_func = nn.AdaptiveMaxPool2d

        self.roi_pool = pool_func(size)

    def forward(self, feature_maps):
        assert feature_maps.dim() == 4, 'Expected 4D input of (N, C, H, W)'
        return self.roi_pool(feature_maps)
