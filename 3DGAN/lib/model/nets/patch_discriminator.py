# ------------------------------------------------------------------------------
# Copyright (c) Tencent
# Licensed under the GPLv3 License.
# Created by Kai Ma (makai0324@gmail.com)
# ------------------------------------------------------------------------------

from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import torch.nn as nn
import numpy as np
import functools


'''
Patch Discriminator
'''
# class NLayer_3D_Discriminator(nn.Module):
#   def __init__(self, input_nc, ndf=64, n_layers=3,
#                norm_layer=nn.BatchNorm3d, use_sigmoid=False):
#     super(NLayer_3D_Discriminator, self).__init__()
#
#     if type(norm_layer) == functools.partial:
#       use_bias = norm_layer.func == nn.InstanceNorm2d
#     else:
#       use_bias = norm_layer == nn.InstanceNorm2d
#
#     kw = 4
#     padw = int(np.ceil((kw - 1.0) / 2))
#
#     sequence = [
#       nn.Conv3d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
#       nn.LeakyReLU(0.2, True)
#     ]
#
#     nf_mult = 1
#     nf_mult_prev = 1
#     for n in range(1, n_layers):
#       nf_mult_prev = nf_mult
#       nf_mult = min(2 ** n, 8)
#       sequence += [
#         nn.Conv3d(ndf * nf_mult_prev, ndf * nf_mult,
#                   kernel_size=kw, stride=2, padding=padw, bias=use_bias),
#         norm_layer(ndf * nf_mult),
#         nn.LeakyReLU(0.2, True)]
#
#     nf_mult_prev = nf_mult
#     nf_mult = min(2 ** n_layers, 8)
#     sequence += [
#       nn.Conv3d(ndf * nf_mult_prev, ndf * nf_mult,
#                 kernel_size=kw, stride=1, padding=padw, bias=use_bias),
#       norm_layer(ndf * nf_mult),
#       nn.LeakyReLU(0.2, True)
#     ]
#
#     sequence += [nn.Conv3d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]
#
#     if use_sigmoid:
#       sequence += [nn.Sigmoid()]
#
#     self.model = nn.Sequential(*sequence)
#
#   def forward(self, input):
#     return self.model(input)


class NLayer_2D_Discriminator(nn.Module):
  def __init__(self, input_nc, ndf=64, n_layers=3,
               norm_layer=nn.BatchNorm2d, use_sigmoid=False, getIntermFeat=False, n_out_channels=1):
    super(NLayer_2D_Discriminator, self).__init__()

    self.getIntermFeat = getIntermFeat
    self.n_layers = n_layers

    if type(norm_layer) == functools.partial:
      use_bias = norm_layer.func == nn.InstanceNorm2d
    else:
      use_bias = norm_layer == nn.InstanceNorm2d

    kw = 4
    padw = int(np.ceil((kw - 1.0) / 2))

    sequence = [[
      nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
      nn.LeakyReLU(0.2, True)
    ]]

    nf_mult = 1
    nf_mult_prev = 1
    for n in range(1, n_layers):
      nf_mult_prev = nf_mult
      nf_mult = min(2 ** n, 8)
      sequence += [[
        nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                  kernel_size=kw, stride=2, padding=padw, bias=use_bias),
        norm_layer(ndf * nf_mult),
        nn.LeakyReLU(0.2, True)]]

    nf_mult_prev = nf_mult
    nf_mult = min(2 ** n_layers, 8)
    sequence += [[
      nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                kernel_size=kw, stride=1, padding=padw, bias=use_bias),
      norm_layer(ndf * nf_mult),
      nn.LeakyReLU(0.2, True)
    ]]

    if use_sigmoid:
      sequence += [[nn.Conv2d(ndf * nf_mult, n_out_channels, kernel_size=kw, stride=1, padding=padw),
                    nn.Sigmoid()]]
    else:
      sequence += [[nn.Conv2d(ndf * nf_mult, n_out_channels, kernel_size=kw, stride=1, padding=padw)]]

    if getIntermFeat:
      for n in range(len(sequence)):
        setattr(self, 'model' + str(n), nn.Sequential(*(sequence[n])))
    else:
      sequence_stream = []
      for n in range(len(sequence)):
        sequence_stream += sequence[n]
      self.model = nn.Sequential(*sequence_stream)

  def forward(self, input):
    if self.getIntermFeat:
      res = [input]
      for n in range(self.n_layers + 2):
        model = getattr(self, 'model' + str(n))
        res.append(model(res[-1]))
      return res[1:]
    else:
      return [self.model(input)]


class NLayer_3D_Discriminator(nn.Module):
  def __init__(self, input_nc, ndf=64, n_layers=3,
               norm_layer=nn.BatchNorm3d, use_sigmoid=False, getIntermFeat=False, n_out_channels=1):
    super(NLayer_3D_Discriminator, self).__init__()

    self.getIntermFeat = getIntermFeat
    self.n_layers = n_layers

    if type(norm_layer) == functools.partial:
      use_bias = norm_layer.func == nn.InstanceNorm2d
    else:
      use_bias = norm_layer == nn.InstanceNorm2d

    kw = 4
    padw = int(np.ceil((kw - 1.0) / 2))

    sequence = [[
      nn.Conv3d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
      nn.LeakyReLU(0.2, True)
    ]]

    nf_mult = 1
    nf_mult_prev = 1
    for n in range(1, n_layers):
      nf_mult_prev = nf_mult
      nf_mult = min(2 ** n, 8)
      sequence += [[
        nn.Conv3d(ndf * nf_mult_prev, ndf * nf_mult,
                  kernel_size=kw, stride=2, padding=padw, bias=use_bias),
        norm_layer(ndf * nf_mult),
        nn.LeakyReLU(0.2, True)]]

    nf_mult_prev = nf_mult
    nf_mult = min(2 ** n_layers, 8)
    sequence += [[
      nn.Conv3d(ndf * nf_mult_prev, ndf * nf_mult,
                kernel_size=kw, stride=1, padding=padw, bias=use_bias),
      norm_layer(ndf * nf_mult),
      nn.LeakyReLU(0.2, True)
    ]]

    if use_sigmoid:
      sequence += [[nn.Conv3d(ndf * nf_mult, n_out_channels, kernel_size=kw, stride=1, padding=padw),
                    nn.Sigmoid()]]
    else:
      sequence += [[nn.Conv3d(ndf * nf_mult, n_out_channels, kernel_size=kw, stride=1, padding=padw)]]

    if getIntermFeat:
      for n in range(len(sequence)):
        setattr(self, 'model' + str(n), nn.Sequential(*(sequence[n])))
    else:
      sequence_stream = []
      for n in range(len(sequence)):
        sequence_stream += sequence[n]
      self.model = nn.Sequential(*sequence_stream)

  def forward(self, input):
    if self.getIntermFeat:
      res = [input]
      for n in range(self.n_layers + 2):
        model = getattr(self, 'model' + str(n))
        res.append(model(res[-1]))
      return res[1:]
    else:
      return [self.model(input)]


'''
Multi-Scale
Patch Discriminator
'''
#############################################################
# 3D Version
#############################################################
class Multiscale_3D_Discriminator(nn.Module):
  def __init__(self, input_nc, ndf=64, n_layers=3,
               norm_layer=nn.BatchNorm2d, use_sigmoid=False,
               getIntermFeat=False, num_D=3, n_out_channels=1):
    super(Multiscale_3D_Discriminator, self).__init__()
    assert num_D >= 1
    self.num_D = num_D
    self.n_layers = n_layers
    self.getIntermFeat = getIntermFeat

    for i in range(num_D):
      netD = NLayer_3D_Discriminator(input_nc, ndf, n_layers, norm_layer, use_sigmoid, getIntermFeat, n_out_channels)
      if getIntermFeat:
        for j in range(n_layers + 2):
          setattr(self, 'scale' + str(i) + '_layer' + str(j), getattr(netD, 'model' + str(j)))
      else:
        setattr(self, 'layer' + str(i), netD.model)

    self.downsample = nn.AvgPool3d(3, stride=2, padding=[1, 1, 1], count_include_pad=False)

  def singleD_forward(self, model, input):
    if self.getIntermFeat:
      result = [input]
      for i in range(len(model)):
        result.append(model[i](result[-1]))
      return result[1:]
    else:
      return [model(input)]

  def forward(self, input):
    num_D = self.num_D
    result = []
    input_downsampled = input
    for i in range(num_D):
      if self.getIntermFeat:
        model = [getattr(self, 'scale' + str(num_D - 1 - i) + '_layer' + str(j)) for j in range(self.n_layers + 2)]
      else:
        model = getattr(self, 'layer' + str(num_D - 1 - i))

      result.append(self.singleD_forward(model, input_downsampled))
      if i != (num_D - 1):
        input_downsampled = self.downsample(input_downsampled)

    return result


#############################################################
# 2D Version
#############################################################
class Multiscale_2D_Discriminator(nn.Module):
  def __init__(self, input_nc, ndf=64, n_layers=3,
               norm_layer=nn.BatchNorm2d, use_sigmoid=False,
               getIntermFeat=False, num_D=3, n_out_channels=1):
    super(Multiscale_2D_Discriminator, self).__init__()
    self.num_D = num_D
    self.n_layers = n_layers
    self.getIntermFeat = getIntermFeat

    for i in range(num_D):
      netD = NLayer_2D_Discriminator(input_nc, ndf, n_layers, norm_layer, use_sigmoid, getIntermFeat, n_out_channels)
      if getIntermFeat:
        for j in range(n_layers + 2):
          setattr(self, 'scale' + str(i) + '_layer' + str(j), getattr(netD, 'model' + str(j)))
      else:
        setattr(self, 'layer' + str(i), netD.model)

    self.downsample = nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)

  def singleD_forward(self, model, input):
    if self.getIntermFeat:
      result = [input]
      for i in range(len(model)):
        result.append(model[i](result[-1]))
      return result[1:]
    else:
      return [model(input)]

  def forward(self, input):
    num_D = self.num_D
    result = []
    input_downsampled = input
    for i in range(num_D):
      if self.getIntermFeat:
        model = [getattr(self, 'scale'+str(num_D-1-i)+'_layer'+str(j)) for j in range(self.n_layers+2)]
      else:
        model = getattr(self, 'layer' + str(num_D - 1 - i))

      result.append(self.singleD_forward(model, input_downsampled))
      if i != (num_D - 1):
        input_downsampled = self.downsample(input_downsampled)

    return result