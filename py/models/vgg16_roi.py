# -*- coding: utf-8 -*-

"""
@date: 2020/3/31 下午2:55
@file: vgg16_roi.py
@author: zj
@description: 
"""

import torch
import torch.nn as nn
import torchvision.models as models

import models.roi_pool as roi_pool


class VGG16_RoI(nn.Module):

    def __init__(self, num_classes=1000, init_weights=True):
        """
        :param num_classes: 类别数，不包括背景类别
        :param init_weights:
        """
        super(VGG16_RoI, self).__init__()
        # VGG16模型的卷积层设置，取消最后一个最大池化层'M'
        feature_list = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512]

        self.features = models.vgg.make_layers(feature_list)
        # self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.roipool = roi_pool.ROI_Pool((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            # nn.Linear(4096, num_classes),
        )
        self.softmax = nn.Linear(4096, num_classes + 1)
        self.bbox = nn.Linear(4096, num_classes * 4)

        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        # x = self.avgpool(x)
        x = self.roipool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        classify = self.softmax(x)
        regression = self.bbox(x)
        return classify, regression

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


if __name__ == '__main__':
    # model = models.vgg16(pretrained=True)
    model = VGG16_RoI()
    print(model)
