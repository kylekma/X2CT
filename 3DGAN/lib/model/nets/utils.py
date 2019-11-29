# ------------------------------------------------------------------------------
# Copyright (c) Tencent
# Licensed under the GPLv3 License.
# Created by Kai Ma (makai0324@gmail.com)
# ------------------------------------------------------------------------------

from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import torch
import torch.nn as nn
import functools
from torch.nn import init
from .doubleBlockLU import DoubleBlockLinearUnit, LinearUnit, SoftLinearUnit

'''
Set Initialization
'''
def init_weights(net, init_type='normal', gain=0.02):
  def init_func(m):
    classname = m.__class__.__name__
    if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
      if init_type == 'normal':
        init.normal_(m.weight.data, 0.0, gain)
      elif init_type == 'xavier':
        init.xavier_normal_(m.weight.data, gain=gain)
      elif init_type == 'kaiming':
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
      elif init_type == 'orthogonal':
        init.orthogonal_(m.weight.data, gain=gain)
      else:
        raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
      if hasattr(m, 'bias') and m.bias is not None:
        init.constant_(m.bias.data, 0.0)
    elif classname.find('BatchNorm2d') != -1:
      init.normal_(m.weight.data, 1.0, gain)
      init.constant_(m.bias.data, 0.0)

  print('initialize network parameters with %s' % init_type)
  net.apply(init_func)

def init_net(net, init_type='normal', gpu_ids=[]):
  if len(gpu_ids) > 0:
    assert (torch.cuda.is_available())
    net.to(gpu_ids[0])
    net = torch.nn.DataParallel(net, gpu_ids)
  init_weights(net, init_type)
  return net

def get_norm_layer(norm_type='instance'):
  if norm_type == 'batch':
    norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
  elif norm_type == 'batch3d':
    norm_layer = functools.partial(nn.BatchNorm3d, affine=True)
  elif norm_type == 'instance':
    norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=True)
  elif norm_type == 'instance3d':
    norm_layer = functools.partial(nn.InstanceNorm3d, affine=False, track_running_stats=True)
  elif norm_type == 'none':
    norm_layer = None
  else:
    raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
  return norm_layer

def get_generator_activation_func(opt, activation_type='relu'):
  if activation_type == 'relu':
    activation_layer = nn.ReLU
  elif activation_type == 'softplus':
    activation_layer = nn.Softplus
  elif activation_type == 'tanh':
    activation_layer = nn.Tanh
  elif activation_type == 'dblu':
    activation_layer = functools.partial(DoubleBlockLinearUnit, low=opt.dblu[0], high=opt.dblu[1], k=opt.dblu[2])
  elif activation_type == 'linear':
    activation_layer = LinearUnit
  elif activation_type == 'softlinear':
    activation_layer = SoftLinearUnit
  else:
    raise NotImplementedError('activation layer [%s] is not found' % activation_type)
  return activation_layer
