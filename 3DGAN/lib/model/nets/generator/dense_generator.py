# ------------------------------------------------------------------------------
# Copyright (c) Tencent
# Licensed under the GPLv3 License.
# Created by Kai Ma (makai0324@gmail.com)
# ------------------------------------------------------------------------------

from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import functools
from lib.model.nets.generator.encoder_decoder_utils import *


def FPNLike_DownStep5(input_shape, encoder_input_channels, decoder_output_channels, decoder_out_activation, encoder_norm_layer, decoder_norm_layer, upsample_mode):
  # 64, 32, 16, 8, 4
  encoder_block_list = [6, 12, 24, 16, 6]
  decoder_block_list = [2, 2, 2, 2, 0]
  growth_rate = 32
  encoder_channel_list = [64]
  decoder_channel_list = [16, 16, 32, 64, 128, 256]
  decoder_begin_size = input_shape // pow(2, len(encoder_block_list))
  return UNetLike_DenseDimensionNet(encoder_input_channels, decoder_output_channels, decoder_begin_size, encoder_block_list, decoder_block_list, growth_rate, encoder_channel_list, decoder_channel_list, decoder_out_activation, encoder_norm_layer, decoder_norm_layer, upsample_mode)

def FPNLike_DownStep5_3(input_shape, encoder_input_channels, decoder_output_channels, decoder_out_activation, encoder_norm_layer, decoder_norm_layer, upsample_mode):
  # 64, 32, 16, 8, 4
  encoder_block_list = [6, 12, 32, 32, 12]
  decoder_block_list = [3, 3, 3, 3, 1]
  growth_rate = 32
  encoder_channel_list = [64]
  decoder_channel_list = [16, 32, 64, 64, 128, 256]
  decoder_begin_size = input_shape // pow(2, len(encoder_block_list))
  return UNetLike_DenseDimensionNet(encoder_input_channels, decoder_output_channels, decoder_begin_size, encoder_block_list, decoder_block_list, growth_rate, encoder_channel_list, decoder_channel_list, decoder_out_activation, encoder_norm_layer, decoder_norm_layer, upsample_mode)

class UNetLike_DenseDimensionNet(nn.Module):
  def __init__(self, encoder_input_channels, decoder_output_channels, decoder_begin_size, encoder_block_list, decoder_block_list, growth_rate, encoder_channel_list, decoder_channel_list, decoder_out_activation, encoder_norm_layer=nn.BatchNorm2d, decoder_norm_layer=nn.BatchNorm3d, upsample_mode='nearest'):
    super(UNetLike_DenseDimensionNet, self).__init__()

    self.decoder_channel_list = decoder_channel_list
    self.n_downsampling = len(encoder_block_list)
    self.decoder_begin_size = decoder_begin_size
    activation = nn.ReLU(True)
    bn_size = 4

    ##############
    # Encoder
    ##############
    if type(encoder_norm_layer) == functools.partial:
      use_bias = encoder_norm_layer.func == nn.InstanceNorm2d
    else:
      use_bias = encoder_norm_layer == nn.InstanceNorm2d

    encoder_layers0 = [
      nn.ReflectionPad2d(3),
      nn.Conv2d(encoder_input_channels, encoder_channel_list[0], kernel_size=7, padding=0, bias=use_bias),
      encoder_norm_layer(encoder_channel_list[0]),
      activation
    ]
    self.encoder_layer = nn.Sequential(*encoder_layers0)

    num_input_channels = encoder_channel_list[0]
    for index, channel in enumerate(encoder_block_list):
      # pooling
      down_layers = [
        encoder_norm_layer(num_input_channels),
        activation,
        nn.Conv2d(num_input_channels, num_input_channels, kernel_size=3, stride=2, padding=1, bias=use_bias),

      ]
      down_layers += [
        Dense_2DBlock(encoder_block_list[index], num_input_channels, bn_size, growth_rate, encoder_norm_layer, activation, use_bias),
      ]
      num_input_channels = num_input_channels + encoder_block_list[index] * growth_rate

      # feature maps are compressed into 1 after the lastest downsample layers
      if index == (self.n_downsampling-1):
        down_layers += [
          nn.AdaptiveAvgPool2d(1)
        ]
      else:
        num_out_channels = num_input_channels // 2
        down_layers += [
          encoder_norm_layer(num_input_channels),
          activation,
          nn.Conv2d(num_input_channels, num_out_channels, kernel_size=1, stride=1, padding=0, bias=use_bias),
        ]
        num_input_channels = num_out_channels
      encoder_channel_list.append(num_input_channels)
      setattr(self, 'encoder_layer' + str(index), nn.Sequential(*down_layers))

    ##############
    # Linker
    ##############
    if type(decoder_norm_layer) == functools.partial:
      use_bias = decoder_norm_layer.func == nn.InstanceNorm3d
    else:
      use_bias = decoder_norm_layer == nn.InstanceNorm3d

    # linker FC
    # apply fc to link 2d and 3d
    self.base_link = nn.Sequential(*[
      nn.Linear(encoder_channel_list[-1], decoder_begin_size**3*decoder_channel_list[-1]),
      nn.Dropout(0.5),
      activation
    ])

    for index, channel in enumerate(encoder_channel_list[:-1]):
      in_channels = channel
      out_channels = decoder_channel_list[index]
      link_layers = [
        Dimension_UpsampleCutBlock(in_channels, out_channels, encoder_norm_layer, decoder_norm_layer, activation, use_bias)
      ]
      setattr(self, 'linker_layer' + str(index), nn.Sequential(*link_layers))

    ##############
    # Decoder
    ##############
    for index, channel in enumerate(decoder_channel_list[:-1]):
      out_channels = channel
      in_channels = decoder_channel_list[index+1]
      decoder_layers = []
      if index != (len(decoder_channel_list) - 2):
        decoder_layers += [
          nn.Conv3d(in_channels * 2, in_channels, kernel_size=3, padding=1, bias=use_bias),
          decoder_norm_layer(in_channels),
          activation
        ]
        for _ in range(decoder_block_list[index]):
          decoder_layers += [
            nn.Conv3d(in_channels, in_channels, kernel_size=3, padding=1, bias=use_bias),
            decoder_norm_layer(in_channels),
            activation
          ]
      decoder_layers += [
        Upsample_3DUnit(3, in_channels, out_channels, decoder_norm_layer, scale_factor=2, upsample_mode=upsample_mode, activation=activation, use_bias=use_bias)
      ]
      setattr(self, 'decoder_layer' + str(index), nn.Sequential(*decoder_layers))

    self.decoder_layer = nn.Sequential(*[
      nn.Conv3d(decoder_channel_list[0], decoder_output_channels, kernel_size=7, padding=3, bias=use_bias),
      decoder_out_activation()
    ])

  def forward(self, input):
    encoder_feature = self.encoder_layer(input)
    next_input = encoder_feature
    for i in range(self.n_downsampling):
      setattr(self, 'feature_linker' + str(i), getattr(self, 'linker_layer' + str(i))(next_input))
      next_input = getattr(self, 'encoder_layer'+str(i))(next_input)

    next_input = self.base_link(next_input.view(next_input.size(0), -1))
    next_input = next_input.view(next_input.size(0), self.decoder_channel_list[-1], self.decoder_begin_size, self.decoder_begin_size, self.decoder_begin_size)

    for i in range(self.n_downsampling - 1, -1, -1):
      if i == (self.n_downsampling - 1):
        next_input = getattr(self, 'decoder_layer' + str(i))(next_input)
      else:
        next_input = getattr(self, 'decoder_layer' + str(i))(torch.cat((next_input, getattr(self, 'feature_linker'+str(i+1))), dim=1))

    return self.decoder_layer(next_input)

