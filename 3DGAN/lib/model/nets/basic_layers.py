# ------------------------------------------------------------------------------
# Copyright (c) Tencent
# Licensed under the GPLv3 License.
# Created by Kai Ma (makai0324@gmail.com)
# ------------------------------------------------------------------------------

from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import torch.nn as nn

'''
Define a resnet block
Network:
  x ->
  -> 3*3 stride=1 padding conv
  -> norm_layer
  -> activation
  -> 3*3 stride=1 padding conv
  -> norm_layer

'''
class ResnetBlock(nn.Module):
  def __init__(self, dim, padding_type, norm_layer,
               activation=nn.ReLU(True), use_dropout=False,
               use_bias=True):
    super(ResnetBlock, self).__init__()
    self.conv_block = self.build_conv_block(dim, padding_type, norm_layer,
                                            activation, use_dropout, use_bias)

  def build_conv_block(self, dim, padding_type, norm_layer,
                       activation, use_dropout, use_bias):
    conv_block = []
    p = 0
    if padding_type == 'reflect':
      conv_block += [nn.ReflectionPad2d(1)]
    elif padding_type == 'replicate':
      conv_block += [nn.ReplicationPad2d(1)]
    elif padding_type == 'zero':
      p = 1
    else:
      raise NotImplementedError('padding [%s] is not implemented' % padding_type)

    conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                   norm_layer(dim),
                   activation]

    p = 0
    if padding_type == 'reflect':
      conv_block += [nn.ReflectionPad2d(1)]
    elif padding_type == 'replicate':
      conv_block += [nn.ReplicationPad2d(1)]
    elif padding_type == 'zero':
      p = 1
    else:
      raise NotImplementedError('padding [%s] is not implemented' % padding_type)

    conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                   norm_layer(dim)]

    return nn.Sequential(*conv_block)

  def forward(self, x):
    out = x + self.conv_block(x)
    return out


'''
Define a resnet block
Network:
  x ->
  -> 1*1 stride=1 padding conv group
  -> norm_layer
  -> activation
  -> 3*3 stride=1 padding conv deepwise
  -> norm_layer
  -> activation
  -> 1*1 stride=1 padding conv group
  -> norm_layer

'''
class ResnetSkip_Block(nn.Module):
  def __init__(self, group, dim, padding_type, norm_layer,
               activation=nn.ReLU(True), use_bias=True):
    super(ResnetSkip_Block, self).__init__()
    self.conv_block = self.build_conv_block(group, dim, padding_type, norm_layer,
                                            activation, use_bias)

  def build_conv_block(self, group, dim, padding_type, norm_layer,
                       activation, use_bias):
    conv_block = []

    conv_block += [nn.Conv2d(dim, dim, kernel_size=1, padding=0, bias=use_bias, groups=group),
                   norm_layer(dim),
                   activation]

    p = 0
    if padding_type == 'reflect':
      conv_block += [nn.ReflectionPad2d(1)]
    elif padding_type == 'replicate':
      conv_block += [nn.ReplicationPad2d(1)]
    elif padding_type == 'zero':
      p = 1
    else:
      raise NotImplementedError('padding [%s] is not implemented' % padding_type)

    conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias, groups=dim),
                   norm_layer(dim),
                   activation]

    conv_block += [nn.Conv2d(dim, dim, kernel_size=1, padding=0, bias=use_bias, groups=group),
                   norm_layer(dim)]

    return nn.Sequential(*conv_block)

  def forward(self, x):
    out = x + self.conv_block(x)
    return out


'''
Define a resnet block
Network:
  x ->
  -> 3*3 stride=1 padding conv
  -> norm_layer
  -> activation
  -> 3*3 stride=1 padding conv
  -> norm_layer

'''
class Resnet_3DBlock(nn.Module):
  def __init__(self, dim, padding_type, norm_layer,
               activation=nn.ReLU(True), use_bias=True):
    super(Resnet_3DBlock, self).__init__()
    self.conv_block = self.build_conv_block(dim, padding_type, norm_layer,
                                            activation, use_bias)

  def build_conv_block(self, dim, padding_type, norm_layer,
                       activation, use_bias):
    conv_block = []
    p = 0
    if padding_type == 'reflect':
      p = 1
    elif padding_type == 'replicate':
      conv_block += [nn.ReplicationPad3d(1)]
    elif padding_type == 'zero':
      p = 1
    else:
      raise NotImplementedError('padding [%s] is not implemented' % padding_type)

    conv_block += [nn.Conv3d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                   norm_layer(dim),
                   activation]

    p = 0
    if padding_type == 'reflect':
      p = 1
    elif padding_type == 'replicate':
      conv_block += [nn.ReplicationPad3d(1)]
    elif padding_type == 'zero':
      p = 1
    else:
      raise NotImplementedError('padding [%s] is not implemented' % padding_type)

    conv_block += [nn.Conv3d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                   norm_layer(dim)]

    return nn.Sequential(*conv_block)

  def forward(self, x):
    out = x + self.conv_block(x)
    return out