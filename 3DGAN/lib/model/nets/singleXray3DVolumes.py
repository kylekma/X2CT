# ------------------------------------------------------------------------------
# Copyright (c) Tencent
# Licensed under the GPLv3 License.
# Created by Kai Ma (makai0324@gmail.com)
# ------------------------------------------------------------------------------

from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import torch.nn as nn
import torch.nn.functional as F


class Basic_layer(nn.Sequential):
  def __init__(self, kernel_size, padding_size, stride, in_channels, out_channels, use_bias):
    super(Basic_layer, self).__init__()
    self.add_module('conv', nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding_size, stride=stride, bias=use_bias))
    self.add_module('bn', nn.BatchNorm2d(out_channels))
    self.add_module('relu', nn.ReLU(True))

  def forward(self, input):
    return super(Basic_layer, self).forward(input)


class ResNet_block(nn.Module):
  def __init__(self, in_channels, out_channels, use_bias):
    super(ResNet_block, self).__init__()
    self.in_channels = in_channels
    self.out_channels = out_channels
    if in_channels == out_channels:
      mid_channels = in_channels // 2
    else:
      mid_channels = in_channels
    self.model = nn.Sequential()
    self.model.add_module('layer1', Basic_layer(kernel_size=1, padding_size=0, stride=1, in_channels=in_channels, out_channels=mid_channels, use_bias=use_bias))
    self.model.add_module('layer2', Basic_layer(kernel_size=3, padding_size=1, stride=1, in_channels=mid_channels,out_channels=mid_channels, use_bias=use_bias))
    self.model.add_module('layer3', Basic_layer(kernel_size=3, padding_size=1, stride=1, in_channels=mid_channels,out_channels=out_channels, use_bias=use_bias))
    if in_channels != out_channels:
      self.skip = Basic_layer(kernel_size=1, padding_size=0, stride=1, in_channels=in_channels, out_channels=out_channels, use_bias=use_bias)

  def forward(self, input):
    if self.in_channels == self.out_channels:
      return F.relu(self.model(input) + input)
    else:
      return F.relu(self.skip(input) + self.model(input))


class Xray3DVolumes_2DModel(nn.Module):
  def __init__(self, output_activation=None):
    super(Xray3DVolumes_2DModel, self).__init__()
    # 128 out
    self.part1 = nn.Sequential(*[
      Basic_layer(kernel_size=7, padding_size=3, stride=2, in_channels=1, out_channels=64, use_bias=False),
      ResNet_block(in_channels=64, out_channels=128, use_bias=False),
      ResNet_block(in_channels=128, out_channels=128, use_bias=False),
      ResNet_block(in_channels=128, out_channels=128, use_bias=False),
      ResNet_block(in_channels=128, out_channels=256, use_bias=False)
    ])
    # 64 out
    self.part2 = nn.Sequential(*[
      nn.MaxPool2d(kernel_size=2, stride=2),
      ResNet_block(in_channels=256, out_channels=256, use_bias=False),
      ResNet_block(in_channels=256, out_channels=256, use_bias=False),
      ResNet_block(in_channels=256, out_channels=256, use_bias=False)
    ])
    # 32 out
    self.part3 = nn.Sequential(*[
      nn.MaxPool2d(kernel_size=2, stride=2),
      ResNet_block(in_channels=256, out_channels=256, use_bias=False),
      ResNet_block(in_channels=256, out_channels=256, use_bias=False),
      ResNet_block(in_channels=256, out_channels=256, use_bias=False)
    ])
    # 16 out
    self.part4 = nn.Sequential(*[
      nn.MaxPool2d(kernel_size=2, stride=2),
      ResNet_block(in_channels=256, out_channels=256, use_bias=False),
      ResNet_block(in_channels=256, out_channels=256, use_bias=False),
      ResNet_block(in_channels=256, out_channels=256, use_bias=False)
    ])
    # 8 in 16 out
    self.part5 = nn.Sequential(*[
      nn.MaxPool2d(kernel_size=2, stride=2),
      ResNet_block(in_channels=256, out_channels=256, use_bias=False),
      ResNet_block(in_channels=256, out_channels=256, use_bias=False),
      ResNet_block(in_channels=256, out_channels=256, use_bias=False),
      ResNet_block(in_channels=256, out_channels=256, use_bias=False),
      ResNet_block(in_channels=256, out_channels=256, use_bias=False),
      nn.ConvTranspose2d(in_channels=256, out_channels=256, kernel_size=4, stride=2, padding=1, groups=256, bias=False)
    ])
    # 16 in 32 out
    self.part6 = nn.Sequential(*[
      ResNet_block(in_channels=256, out_channels=256, use_bias=False),
      nn.ConvTranspose2d(in_channels=256, out_channels=256, kernel_size=4, stride=2, padding=1, groups=256, bias=False)
    ])
    # 32 in 64 out
    self.part7 = nn.Sequential(*[
      ResNet_block(in_channels=256, out_channels=256, use_bias=False),
      nn.ConvTranspose2d(in_channels=256, out_channels=256, kernel_size=4, stride=2, padding=1, groups=256, bias=False)
    ])
    # 64 in 128 out
    self.part8 = nn.Sequential(*[
      ResNet_block(in_channels=256, out_channels=256, use_bias=False),
      nn.ConvTranspose2d(in_channels=256, out_channels=256, kernel_size=4, stride=2, padding=1, groups=256, bias=False)
    ])
    # 128
    self.part9 = nn.Sequential(*[
      Basic_layer(kernel_size=1, padding_size=0, stride=1, in_channels=256, out_channels=256, use_bias=False),
      nn.Conv2d(in_channels=256, out_channels=128, kernel_size=1, stride=1, padding=0, bias=False)
    ])
    if output_activation is not None:
      self.part9.add_module('out_activation', output_activation(True))

  def forward(self, input):
    feature128 = self.part1(input)
    feature64 = self.part2(feature128)
    feature32 = self.part3(feature64)
    feature16 = self.part4(feature32)
    feature16up = self.part5(feature16)
    feature32up = self.part6(feature16 + feature16up)
    feature64up = self.part7(feature32 + feature32up)
    feature128up = self.part8(feature64 + feature64up)
    output = self.part9(feature128 + feature128up)
    return output