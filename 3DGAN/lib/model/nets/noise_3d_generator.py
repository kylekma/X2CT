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


class Resnet_3DGenerator(nn.Module):
  def __init__(self, noise_len, input_shape=(4,4,4), input_nc=512, output_shape=(128,128,128), output_nc=1,
               norm_layer=nn.BatchNorm3d, upsample_mode='nearest', n_blocks=3, padding_type='reflect'):
    super(Resnet_3DGenerator, self).__init__()
    assert(n_blocks >= 0)
    assert (input_shape[0] == input_shape[1]) and (input_shape[1] == input_shape[2])
    assert (output_shape[0] == output_shape[1]) and (output_shape[1] == output_shape[2])

    self.input_shape = input_shape
    self.input_nc = input_nc

    if type(norm_layer) == functools.partial:
      use_bias = norm_layer.func == nn.InstanceNorm3d
    else:
      use_bias = norm_layer == nn.InstanceNorm3d

    activation = nn.ReLU(True)

    aim_len = functools.reduce(lambda x,y:x*y, input_shape) * input_nc
    self.fc = nn.Sequential(*[nn.Linear(noise_len, aim_len),
                             activation])

    model = [
      self.build_upsampling_block(input_nc, input_nc, norm_layer, activation, up_sample=False,
                                  padding_type=padding_type, upsample_mode=upsample_mode, block_n=n_blocks, use_bias=use_bias)
    ]

    # #upsample
    max_up = int(np.max(np.log2(np.asarray(output_shape) / np.asarray(input_shape))))
    max_down = int(np.log2(input_nc/output_nc))
    assert max_up < max_down, 'upsampling overstep border'

    out_channel = input_nc
    for i in range(max_up-1):
      in_channel = out_channel
      out_channel = int(in_channel / 2)
      model += [
        self.build_upsampling_block(in_channel, out_channel, norm_layer, activation, up_sample=True,
                                    padding_type=padding_type, upsample_mode=upsample_mode, block_n=n_blocks,
                                    use_bias=use_bias)
      ]

    # last sampling
    in_channel = out_channel
    out_channel = int(in_channel / 2)
    model += [
      self.build_upsampling_block(in_channel, out_channel, norm_layer, activation, up_sample=True, up_conv=False,
                                  padding_type=padding_type, upsample_mode=upsample_mode, block_n=n_blocks,
                                  use_bias=use_bias)
    ]

    p = 0
    if padding_type == 'reflect':
      p = 1
    elif padding_type == 'replicate':
      model += [nn.ReplicationPad3d(1)]
    elif padding_type == 'zero':
      p = 1
    else:
      raise NotImplementedError('padding [%s] is not implemented' % padding_type)
    model += [
      nn.Conv3d(out_channel, output_nc, kernel_size=3, padding=p, bias=use_bias),
      norm_layer(output_nc),
      # nn.Tanh()
      # nn.Sigmoid()
      activation
    ]

    self.model = nn.Sequential(*model)

  def build_upsampling_block(self, input_nc, output_nc, norm_layer, activation, up_sample=True, up_conv=True,
                       padding_type='reflect', upsample_mode='nearest', block_n=3, use_bias=True):
    blocks = []
    if up_sample:
      # #upsample and convolution
      blocks += [
        nn.Upsample(scale_factor=2, mode=upsample_mode)
      ]

      p = 0
      if padding_type == 'reflect':
        p = 1
      elif padding_type == 'replicate':
        blocks += [nn.ReplicationPad3d(1)]
      elif padding_type == 'zero':
        p = 1
      else:
        raise NotImplementedError('padding [%s] is not implemented' % padding_type)

      blocks += [
        nn.Conv3d(input_nc, output_nc, kernel_size=3, padding=p, bias=use_bias),
        norm_layer(output_nc),
        activation
      ]

    if up_conv:
      for i in range(block_n):
        blocks +=[
          Resnet_3DBlock(output_nc, padding_type=padding_type, activation=activation, norm_layer=norm_layer,
                         use_dropout=False, use_bias=use_bias),
          activation
        ]

    return nn.Sequential(*blocks)

  def forward(self, input):
    linear_tensor = self.fc(input)
    # n 1 D H W
    init_3d_tensor = linear_tensor.view((linear_tensor.size(0), self.input_nc, *self.input_shape))
    # #upsample and convolution
    return self.model(init_3d_tensor)


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
               activation=nn.ReLU(True), use_dropout=False,
               use_bias=True):
    super(Resnet_3DBlock, self).__init__()
    self.conv_block = self.build_conv_block(dim, padding_type, norm_layer,
                                            activation, use_dropout, use_bias)

  def build_conv_block(self, dim, padding_type, norm_layer,
                       activation, use_dropout, use_bias):
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

    if use_dropout:
      conv_block += [nn.Dropout(0.5)]

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