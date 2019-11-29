# ------------------------------------------------------------------------------
# Copyright (c) Tencent
# Licensed under the GPLv3 License.
# Created by Kai Ma (makai0324@gmail.com)
# ------------------------------------------------------------------------------

from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import torch
import numpy as np
import functools

from lib.model.nets.generator.encoder_decoder_utils import *


'''
Wrapping Architecture
'''
class Link_Encoder_Decoder(nn.Module):
  def __init__(self, encoder, decoder, linker):
    super(Link_Encoder_Decoder, self).__init__()
    self.encoder = encoder
    self.decoder = decoder
    self.linker = linker

  def forward(self, input):
    return self.decoder(self.linker(self.encoder(input)))


'''
Link Encoder to Decoder
'''
class Linker_FC(nn.Module):
  def __init__(self, noise_len, encoder_out_channel, encoder_out_shape, decoder_input_shape, decoder_input_nc):
    super(Linker_FC, self).__init__()
    activation = nn.ReLU(True)
    # Encoder
    encoder_input_cell = encoder_out_shape[0] * encoder_out_shape[1] * encoder_out_channel
    encoder_fc = [
      nn.Linear(int(encoder_input_cell), noise_len),
      activation]
    # Decoder
    self.decoder_input_nc = decoder_input_nc
    self.decoder_input_shape = decoder_input_shape
    decoder_output_cell = functools.reduce(lambda x,y:x*y, decoder_input_shape) * decoder_input_nc
    decoder_fc = [
      nn.Linear(noise_len, decoder_output_cell),
      activation
    ]
    print('Link {} to {} to {}'.format(encoder_input_cell, noise_len, decoder_output_cell))
    self.linker = nn.Sequential(*(encoder_fc+decoder_fc))

  def forward(self, input):
    return self.linker(input.view(input.size(0), -1)).view(input.size(0), self.decoder_input_nc, *self.decoder_input_shape)

class Linker_DimensionUp_Conv(nn.Module):
  def __init__(self, encoder_out_channel, encoder_out_shape, decoder_input_shape, decoder_input_nc, norm_layer):
    super(Linker_DimensionUp_Conv, self).__init__()
    if type(norm_layer) == functools.partial:
      use_bias = norm_layer.func == nn.InstanceNorm2d
    else:
      use_bias = norm_layer == nn.InstanceNorm2d

    activation = nn.ReLU(True)

    self.linker = Dimension_UpsampleBlock(encoder_out_channel, decoder_input_nc, norm_layer, activation, use_bias)
    print('Link shape={}{} to shape={}{}'.format(encoder_out_shape, encoder_out_channel, decoder_input_shape, decoder_input_nc))

  def forward(self, input):
    return self.linker(input)


class Linker_FC_multiview(nn.Module):
  def __init__(self, input_views_n, noise_len, encoder_out_channel, encoder_out_shape, decoder_input_shape, decoder_input_nc):
    super(Linker_FC_multiview, self).__init__()
    self.input_views_n = input_views_n
    activation = nn.ReLU(True)
    # Encoder
    encoder_input_cell = encoder_out_shape[0] * encoder_out_shape[1] * encoder_out_channel
    for i in range(input_views_n):
      encoder_fc = [
        nn.Linear(int(encoder_input_cell), noise_len),
        activation]
      setattr(self, 'view' + str(i), nn.Sequential(*(encoder_fc)))
    # Fuse module
    fuse_module = [
      nn.Linear(input_views_n * noise_len, noise_len),
      activation]
    self.fuse_module = nn.Sequential(*fuse_module)
    # Decoder
    self.decoder_input_nc = decoder_input_nc
    self.decoder_input_shape = decoder_input_shape
    decoder_output_cell = functools.reduce(lambda x,y:x*y, decoder_input_shape) * decoder_input_nc
    decoder_fc = [
      nn.Linear(noise_len, decoder_output_cell),
      activation
    ]
    self.decoder_fc = nn.Sequential(*(decoder_fc))
    print('Link {} to {} to {} to {}'.format(encoder_input_cell, 2*noise_len, noise_len, decoder_output_cell))

  def forward(self, input):
    assert len(input) == self.input_views_n
    out_list = []
    for index, input_view in enumerate(input):
      out = getattr(self, 'view' + str(index))(input_view.view(input_view.size(0), -1))
      out_list.append(out)
    return self.decoder_fc(self.fuse_module(torch.cat(out_list, dim=1))).view(input_view.size(0), self.decoder_input_nc, *self.decoder_input_shape)


'''
Encoder
'''
class Conv_connect_2D_encoder(nn.Module):
  def __init__(self, input_shape=(128, 128), input_nc=1, ngf=16, norm_layer=nn.BatchNorm2d,
               n_downsampling=4, n_blocks=3):
    super(Conv_connect_2D_encoder, self).__init__()
    assert (n_blocks >= 0)

    if type(norm_layer) == functools.partial:
      use_bias = norm_layer.func == nn.InstanceNorm2d
    else:
      use_bias = norm_layer == nn.InstanceNorm2d

    activation = nn.ReLU(True)

    model = []

    out_c = input_nc
    for i in range(2):
      in_c = out_c
      out_c = 3
      model += [nn.ReflectionPad2d(1),
               nn.Conv2d(in_c, out_c, kernel_size=3, padding=0, bias=use_bias),
               norm_layer(out_c),
               activation]
    model += [nn.ReflectionPad2d(1),
              nn.Conv2d(out_c, ngf, kernel_size=3, padding=0, stride=2, bias=use_bias),
              norm_layer(ngf),
              activation]

    ## downsample
    for i in range(n_downsampling-1):
      mult = 2 ** i
      for _ in range(n_blocks):
        model += [ResnetBlock(ngf * mult, activation=activation, norm_layer=norm_layer, use_bias=use_bias),
                  activation]
      model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1),
                norm_layer(ngf * mult * 2),
                activation]
    self.next_input_channels = ngf * 2 ** (n_downsampling - 1)
    encoder_stride = 2 ** n_downsampling
    self.next_feature_size = (input_shape[0] // encoder_stride, input_shape[1] // encoder_stride)
    self.model = nn.Sequential(*model)

  @property
  def OutChannels(self):
    return self.next_input_channels

  @property
  def OutFeatureSize(self):
    return self.next_feature_size

  def forward(self, input):
    return self.model(input)

class DenseConv_connect_2D_encoder(nn.Module):
  def __init__(self, input_shape=(128, 128), input_nc=1, ngf=16, norm_layer=nn.BatchNorm2d, n_downsampling=4, n_blocks=3):
    super(DenseConv_connect_2D_encoder, self).__init__()
    assert (n_blocks >= 0)

    if type(norm_layer) == functools.partial:
      use_bias = norm_layer.func == nn.InstanceNorm2d
    else:
      use_bias = norm_layer == nn.InstanceNorm2d

    activation = nn.ReLU(True)

    model = []

    out_c = input_nc
    for i in range(2):
      in_c = out_c
      out_c = 3
      model += [nn.ReflectionPad2d(1),
               nn.Conv2d(in_c, out_c, kernel_size=3, padding=0, bias=use_bias),
               norm_layer(out_c),
               activation]
    model += [nn.ReflectionPad2d(1),
              nn.Conv2d(out_c, ngf, kernel_size=3, padding=0, stride=2, bias=use_bias),
              norm_layer(ngf),
              activation]

    num_layers = 5
    growth_rate = 16
    bn_size = 4
    num_input_channels = ngf
    ## downsample
    for i in range(n_downsampling):
      model += [
        Dense_2DBlock(num_layers, num_input_channels, bn_size, growth_rate, norm_layer, activation, use_bias)
      ]
      num_input_channels = num_input_channels + num_layers*growth_rate
      if i != n_downsampling-1:
        num_out_channels = num_input_channels // 2
        model += [
          DenseBlock2D_Transition(num_input_channels, num_out_channels, norm_layer, activation, use_bias)
        ]
        num_input_channels = num_out_channels
    model += [
      norm_layer(num_input_channels)
    ]
    self.next_input_channels = num_input_channels
    encoder_stride = 2 ** n_downsampling
    self.next_feature_size = (input_shape[0] // encoder_stride, input_shape[1] // encoder_stride)
    self.model = nn.Sequential(*model)

  @property
  def OutChannels(self):
    return self.next_input_channels

  @property
  def OutFeatureSize(self):
    return self.next_feature_size

  def forward(self, input):
    return self.model(input)


class Conv_connect_2D_encoder_multiview(nn.Module):
  def __init__(self, input_views, input_shape=(128, 128), input_nc=1, ngf=16, norm_layer=nn.BatchNorm2d, n_downsampling=4, n_blocks=3):
    super(Conv_connect_2D_encoder_multiview, self).__init__()

    self.input_views = input_views

    for i in range(input_views):
      layer = Conv_connect_2D_encoder(input_shape, input_nc, ngf, norm_layer, n_downsampling, n_blocks)
      setattr(self, 'view'+str(i), layer)
      self.next_input_channels = layer.OutChannels
      self.next_feature_size = layer.OutFeatureSize

  @property
  def OutChannels(self):
    return self.next_input_channels

  @property
  def OutFeatureSize(self):
    return self.next_feature_size

  def forward(self, input):
    assert len(input) == self.input_views
    out_list = []
    for index, view in enumerate(input):
      out = getattr(self, 'view'+str(index))(view)
      out_list.append(out)
    return out_list

class DenseConv_connect_2D_encoder_multiview(nn.Module):
  def __init__(self, input_views, input_shape=(128, 128), input_nc=1, ngf=16, norm_layer=nn.BatchNorm2d, n_downsampling=4, n_blocks=3):
    super(DenseConv_connect_2D_encoder_multiview, self).__init__()

    self.input_views = input_views

    for i in range(input_views):
      layer = DenseConv_connect_2D_encoder(input_shape, input_nc, ngf, norm_layer, n_downsampling, n_blocks)
      setattr(self, 'view'+str(i), layer)
      self.next_input_channels = layer.OutChannels
      self.next_feature_size = layer.OutFeatureSize

  @property
  def OutChannels(self):
    return self.next_input_channels

  @property
  def OutFeatureSize(self):
    return self.next_feature_size

  def forward(self, input):
    assert len(input) == self.input_views
    out_list = []
    for index, view in enumerate(input):
      out = getattr(self, 'view'+str(index))(view)
      out_list.append(out)
    return out_list

########################################
'''
Decoder
'''
########################################
'''
3DGenerator_Decoder
=> Decoder + out_activation
'''
class Generator_3DResNetDecoder(nn.Module):
  def __init__(self, input_shape=(4,4,4), input_nc=512, output_shape=(128,128,128), output_nc=1, norm_layer=nn.BatchNorm3d, upsample_mode='nearest', n_blocks=3, out_activation=nn.ReLU):
    super(Generator_3DResNetDecoder, self).__init__()
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

    model = [
      self.build_upsampling_block(input_nc, input_nc, norm_layer, activation, up_sample=False, upsample_mode=upsample_mode, block_n=n_blocks, use_bias=use_bias)
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
        self.build_upsampling_block(in_channel, out_channel, norm_layer, activation, up_sample=True, upsample_mode=upsample_mode, block_n=n_blocks, use_bias=use_bias)
      ]

    # last sampling
    in_channel = out_channel
    out_channel = int(in_channel / 2)
    model += [
      self.build_upsampling_block(in_channel, out_channel, norm_layer, activation, up_sample=True, up_conv=False, upsample_mode=upsample_mode, block_n=n_blocks, use_bias=use_bias)
    ]

    model += [
      nn.Conv3d(out_channel, output_nc, kernel_size=3, padding=1, bias=use_bias),
    ]

    # out activation
    model += [
      out_activation()
    ]

    self.model = nn.Sequential(*model)

  def build_upsampling_block(self, input_nc, output_nc, norm_layer, activation, up_sample=True, up_conv=True,upsample_mode='nearest', block_n=3, use_bias=True):
    blocks = []
    if up_sample:
      # #upsample and convolution
      blocks += [
        nn.Upsample(scale_factor=2, mode=upsample_mode)
      ]

      blocks += [
        nn.Conv3d(input_nc, output_nc, kernel_size=3, padding=1, bias=use_bias),
        norm_layer(output_nc),
        activation
      ]

    if up_conv:
      for i in range(block_n):
        blocks += [
          Resnet_3DBlock(output_nc, activation=activation, norm_layer=norm_layer, use_bias=use_bias),
          activation
        ]

    return nn.Sequential(*blocks)

  def forward(self, input):
    # #upsample and convolution
    return self.model(input)


class Generator_TransposedConvDecoder(nn.Module):
  def __init__(self, input_shape=(4,4,4), input_nc=512, output_shape=(128,128,128), output_nc=1, norm_layer=nn.BatchNorm3d, upsample_mode='nearest', n_blocks=3, out_activation=nn.ReLU):
    super(Generator_TransposedConvDecoder, self).__init__()
    assert(n_blocks >= 0)
    assert (input_shape[0] == input_shape[1]) and (input_shape[1] == input_shape[2])
    assert (output_shape[0] == output_shape[1]) and (output_shape[1] == output_shape[2])

    self.input_shape = input_shape
    self.input_nc = input_nc
    self.minimal_nc = 16

    if type(norm_layer) == functools.partial:
      use_bias = norm_layer.func == nn.InstanceNorm3d
    else:
      use_bias = norm_layer == nn.InstanceNorm3d

    activation = nn.ReLU(True)

    model = [
      self.build_upsampling_block(input_nc, input_nc, norm_layer, activation, up_sample=False, upsample_mode=upsample_mode, block_n=n_blocks, use_bias=use_bias)
    ]

    # #upsample
    max_up = int(np.max(np.log2(np.asarray(output_shape) / np.asarray(input_shape))))
    max_down = int(np.log2(input_nc/output_nc))
    assert max_up < max_down, 'upsampling overstep border'

    out_channel = input_nc
    for i in range(max_up-1):
      in_channel = out_channel
      out_channel = int(in_channel / 2) if (int(in_channel / 2) > self.minimal_nc) else self.minimal_nc
      model += [
        self.build_upsampling_block(in_channel, out_channel, norm_layer, activation, up_sample=True, upsample_mode=upsample_mode, block_n=n_blocks, use_bias=use_bias)
      ]

    # last sampling
    in_channel = out_channel
    out_channel = int(in_channel / 2)
    model += [
      self.build_upsampling_block(in_channel, out_channel, norm_layer, activation, up_sample=True, up_conv=False, upsample_mode=upsample_mode, block_n=n_blocks, use_bias=use_bias)
    ]

    model += [
      nn.Conv3d(out_channel, output_nc, kernel_size=3, padding=1, bias=use_bias),
    ]

    # out activation
    model += [
      out_activation()
    ]

    self.model = nn.Sequential(*model)

  def build_upsampling_block(self, input_nc, output_nc, norm_layer, activation, up_sample=True, up_conv=True, upsample_mode='nearest', block_n=3, use_bias=True):
    blocks = []
    if up_sample:
      # #upsample and convolution
      blocks += [
        Upsample_3DUnit(3, input_nc, output_nc, norm_layer, scale_factor=2, upsample_mode=upsample_mode, activation=activation, use_bias=use_bias)
      ]

    if up_conv:
      for i in range(block_n):
        blocks += [
          Resnet_3DBlock(output_nc, activation=activation, norm_layer=norm_layer, use_bias=use_bias),
          activation
        ]

    return nn.Sequential(*blocks)

  def forward(self, input):
    # #upsample and convolution
    return self.model(input)


class Generator_3DNormResNetDecoder(nn.Module):
  def __init__(self, input_shape=(4,4,4), input_nc=512, output_shape=(128,128,128), output_nc=1, norm_layer=nn.BatchNorm3d, upsample_mode='nearest', n_blocks=3, out_activation=nn.ReLU):
    super(Generator_3DNormResNetDecoder, self).__init__()
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

    model = [
      self.build_upsampling_block(input_nc, input_nc, norm_layer, activation, up_sample=False, upsample_mode=upsample_mode, block_n=n_blocks, use_bias=use_bias)
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
        self.build_upsampling_block(in_channel, out_channel, norm_layer, activation, up_sample=True, upsample_mode=upsample_mode, block_n=n_blocks, use_bias=use_bias)
      ]

    # last sampling
    in_channel = out_channel
    out_channel = int(in_channel / 2)
    model += [
      self.build_upsampling_block(in_channel, out_channel, norm_layer, activation, up_sample=True, up_conv=False, upsample_mode=upsample_mode, block_n=n_blocks, use_bias=use_bias)
    ]

    model += [
      nn.Conv3d(out_channel, output_nc, kernel_size=3, padding=1, bias=use_bias),
      norm_layer(output_nc)
    ]

    # out activation
    model += [
      out_activation()
    ]

    self.model = nn.Sequential(*model)

  def build_upsampling_block(self, input_nc, output_nc, norm_layer, activation, up_sample=True, up_conv=True,upsample_mode='nearest', block_n=3, use_bias=True):
    blocks = []
    if up_sample:
      # #upsample and convolution
      blocks += [
        nn.Upsample(scale_factor=2, mode=upsample_mode)
      ]

      blocks += [
        nn.Conv3d(input_nc, output_nc, kernel_size=3, padding=1, bias=use_bias),
        norm_layer(output_nc),
        activation
      ]

    if up_conv:
      for i in range(block_n):
        blocks += [
          Resnet_3DBlock(output_nc, activation=activation, norm_layer=norm_layer, use_bias=use_bias),
          activation
        ]

    return nn.Sequential(*blocks)

  def forward(self, input):
    # #upsample and convolution
    return self.model(input)


'''
3DGenerator_Decoder
=> Decoder + out_activation
'''
class Generator_3DLinearResNetDecoder(nn.Module):
  def __init__(self, input_shape=(4,4,4), input_nc=512, output_shape=(128,128,128), output_nc=1, norm_layer=nn.BatchNorm3d, upsample_mode='nearest', n_blocks=3, out_activation=nn.ReLU):
    super(Generator_3DLinearResNetDecoder, self).__init__()
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

    model = [
      self.build_upsampling_block(input_nc, input_nc, norm_layer, activation, up_sample=False, upsample_mode=upsample_mode, block_n=n_blocks, use_bias=use_bias)
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
        self.build_upsampling_block(in_channel, out_channel, norm_layer, activation, up_sample=True, upsample_mode=upsample_mode, block_n=n_blocks, use_bias=use_bias)
      ]

    # last sampling
    in_channel = out_channel
    out_channel = int(in_channel / 2)
    model += [
      self.build_upsampling_block(in_channel, out_channel, norm_layer, activation, up_sample=True, up_conv=False, upsample_mode=upsample_mode, block_n=n_blocks, use_bias=use_bias)
    ]

    model += [
      nn.Conv3d(out_channel, output_nc, kernel_size=3, padding=1, bias=use_bias),
    ]

    # out activation
    model += [
      out_activation()
    ]

    self.model = nn.Sequential(*model)

  def build_upsampling_block(self, input_nc, output_nc, norm_layer, activation, up_sample=True, up_conv=True,upsample_mode='nearest', block_n=3, use_bias=True):
    blocks = []
    if up_sample:
      # #upsample and convolution
      blocks += [
        nn.Upsample(scale_factor=2, mode=upsample_mode)
      ]

      blocks += [
        nn.Conv3d(input_nc, output_nc, kernel_size=3, padding=1, bias=use_bias),
        norm_layer(output_nc),
        activation
      ]

    if up_conv:
      for i in range(block_n):
        blocks += [
          Resnet_3DBlock(output_nc, activation=activation, norm_layer=norm_layer, use_bias=use_bias)
        ]

    return nn.Sequential(*blocks)

  def forward(self, input):
    # #upsample and convolution
    return self.model(input)


'''
3DGenerator_Decoder
=> Decoder + out_activation
'''
class Generator_3DLinearSTExpand2ResNetDecoder(nn.Module):
  def __init__(self, input_shape=(4,4,4), input_nc=512, output_shape=(128,128,128), output_nc=1, norm_layer=nn.BatchNorm3d, upsample_mode='nearest', n_blocks=3, out_activation=nn.ReLU):
    super(Generator_3DLinearSTExpand2ResNetDecoder, self).__init__()
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

    model = [
      self.build_upsampling_block(input_nc, input_nc, norm_layer, activation, up_sample=False, upsample_mode=upsample_mode, block_n=n_blocks, use_bias=use_bias)
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
        self.build_upsampling_block(in_channel, out_channel, norm_layer, activation, up_sample=True, upsample_mode=upsample_mode, block_n=n_blocks, use_bias=use_bias)
      ]

    # last sampling
    in_channel = out_channel
    out_channel = int(in_channel / 2)
    model += [
      self.build_upsampling_block(in_channel, out_channel, norm_layer, activation, up_sample=True, up_conv=False, upsample_mode=upsample_mode, block_n=n_blocks, use_bias=use_bias)
    ]

    model += [
      nn.Conv3d(out_channel, output_nc, kernel_size=3, padding=1, bias=use_bias),
    ]

    # out activation
    model += [
      out_activation()
    ]

    self.model = nn.Sequential(*model)

  def build_upsampling_block(self, input_nc, output_nc, norm_layer, activation, up_sample=True, up_conv=True,upsample_mode='nearest', block_n=3, use_bias=True):
    blocks = []
    if up_sample:
      # #upsample and convolution
      blocks += [
        nn.Upsample(scale_factor=2, mode=upsample_mode)
      ]

      blocks += [
        nn.Conv3d(input_nc, output_nc, kernel_size=3, padding=1, bias=use_bias),
        norm_layer(output_nc),
        activation
      ]

    if up_conv:
      for i in range(block_n):
        blocks += [
          Resnet_ST3DBlock(output_nc, 2, activation=activation, norm_layer=norm_layer, use_bias=use_bias)
        ]

    return nn.Sequential(*blocks)

  def forward(self, input):
    # #upsample and convolution
    return self.model(input)

class Generator_3DLinearSTShink2ResNetDecoder(nn.Module):
  def __init__(self, input_shape=(4,4,4), input_nc=512, output_shape=(128,128,128), output_nc=1, norm_layer=nn.BatchNorm3d, upsample_mode='nearest', n_blocks=3, out_activation=nn.ReLU):
    super(Generator_3DLinearSTShink2ResNetDecoder, self).__init__()
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

    model = [
      self.build_upsampling_block(input_nc, input_nc, norm_layer, activation, up_sample=False, upsample_mode=upsample_mode, block_n=n_blocks, use_bias=use_bias)
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
        self.build_upsampling_block(in_channel, out_channel, norm_layer, activation, up_sample=True, upsample_mode=upsample_mode, block_n=n_blocks, use_bias=use_bias)
      ]

    # last sampling
    in_channel = out_channel
    out_channel = int(in_channel / 2)
    model += [
      self.build_upsampling_block(in_channel, out_channel, norm_layer, activation, up_sample=True, up_conv=False, upsample_mode=upsample_mode, block_n=n_blocks, use_bias=use_bias)
    ]

    model += [
      nn.Conv3d(out_channel, output_nc, kernel_size=3, padding=1, bias=use_bias),
    ]

    # out activation
    model += [
      out_activation()
    ]

    self.model = nn.Sequential(*model)

  def build_upsampling_block(self, input_nc, output_nc, norm_layer, activation, up_sample=True, up_conv=True,upsample_mode='nearest', block_n=3, use_bias=True):
    blocks = []
    if up_sample:
      # #upsample and convolution
      blocks += [
        nn.Upsample(scale_factor=2, mode=upsample_mode)
      ]

      blocks += [
        nn.Conv3d(input_nc, output_nc, kernel_size=3, padding=1, bias=use_bias),
        norm_layer(output_nc),
        activation
      ]

    if up_conv:
      for i in range(block_n):
        blocks += [
          Resnet_ST3DBlock(output_nc, 0.5, activation=activation, norm_layer=norm_layer, use_bias=use_bias)
        ]

    return nn.Sequential(*blocks)

  def forward(self, input):
    # #upsample and convolution
    return self.model(input)

class Generator_3DLinearSTResNetDecoder(nn.Module):
  def __init__(self, input_shape=(4,4,4), input_nc=512, output_shape=(128,128,128), output_nc=1, norm_layer=nn.BatchNorm3d, upsample_mode='nearest', n_blocks=3, out_activation=nn.ReLU):
    super(Generator_3DLinearSTResNetDecoder, self).__init__()
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

    model = [
      self.build_upsampling_block(input_nc, input_nc, norm_layer, activation, up_sample=False, upsample_mode=upsample_mode, block_n=n_blocks, use_bias=use_bias)
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
        self.build_upsampling_block(in_channel, out_channel, norm_layer, activation, up_sample=True, upsample_mode=upsample_mode, block_n=n_blocks, use_bias=use_bias)
      ]

    # last sampling
    in_channel = out_channel
    out_channel = int(in_channel / 2)
    model += [
      self.build_upsampling_block(in_channel, out_channel, norm_layer, activation, up_sample=True, up_conv=False, upsample_mode=upsample_mode, block_n=n_blocks, use_bias=use_bias)
    ]

    model += [
      nn.Conv3d(out_channel, output_nc, kernel_size=3, padding=1, bias=use_bias),
    ]

    # out activation
    model += [
      out_activation()
    ]

    self.model = nn.Sequential(*model)

  def build_upsampling_block(self, input_nc, output_nc, norm_layer, activation, up_sample=True, up_conv=True,upsample_mode='nearest', block_n=3, use_bias=True):
    blocks = []
    if up_sample:
      # #upsample and convolution
      blocks += [
        nn.Upsample(scale_factor=2, mode=upsample_mode)
      ]

      blocks += [
        nn.Conv3d(input_nc, output_nc, kernel_size=3, padding=1, bias=use_bias),
        norm_layer(output_nc),
        activation
      ]

    if up_conv:
      for i in range(block_n):
        blocks += [
          Resnet_ST3DBlock(output_nc, 1, activation=activation, norm_layer=norm_layer, use_bias=use_bias)
        ]

    return nn.Sequential(*blocks)

  def forward(self, input):
    # #upsample and convolution
    return self.model(input)

class Generator_3DSTResNetDecoder(nn.Module):
  def __init__(self, input_shape=(4,4,4), input_nc=512, output_shape=(128,128,128), output_nc=1, norm_layer=nn.BatchNorm3d, upsample_mode='nearest', n_blocks=3, out_activation=nn.ReLU):
    super(Generator_3DSTResNetDecoder, self).__init__()
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

    model = [
      self.build_upsampling_block(input_nc, input_nc, norm_layer, activation, up_sample=False, upsample_mode=upsample_mode, block_n=n_blocks, use_bias=use_bias)
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
        self.build_upsampling_block(in_channel, out_channel, norm_layer, activation, up_sample=True, upsample_mode=upsample_mode, block_n=n_blocks, use_bias=use_bias)
      ]

    # last sampling
    in_channel = out_channel
    out_channel = int(in_channel / 2)
    model += [
      self.build_upsampling_block(in_channel, out_channel, norm_layer, activation, up_sample=True, up_conv=False, upsample_mode=upsample_mode, block_n=n_blocks, use_bias=use_bias)
    ]

    model += [
      nn.Conv3d(out_channel, output_nc, kernel_size=3, padding=1, bias=use_bias)
    ]

    # out activation
    model += [
      out_activation()
    ]

    self.model = nn.Sequential(*model)

  def build_upsampling_block(self, input_nc, output_nc, norm_layer, activation, up_sample=True, up_conv=True,upsample_mode='nearest', block_n=3, use_bias=True):
    blocks = []
    if up_sample:
      # #upsample and convolution
      blocks += [
        nn.Upsample(scale_factor=2, mode=upsample_mode)
      ]

      blocks += [
        nn.Conv3d(input_nc, output_nc, kernel_size=3, padding=1, bias=use_bias),
        norm_layer(output_nc),
        activation
      ]

    if up_conv:
      for i in range(block_n):
        blocks += [
          Resnet_ST3DBlock(output_nc, 1, activation=activation, norm_layer=norm_layer, use_bias=use_bias),
          activation
        ]

    return nn.Sequential(*blocks)

  def forward(self, input):
    # #upsample and convolution
    return self.model(input)

class Generator_3DDenseNetDecoder(nn.Module):
  def __init__(self, input_shape=(4,4,4), input_nc=512, output_shape=(128,128,128), output_nc=1, norm_layer=nn.BatchNorm3d, upsample_mode='nearest', n_blocks=3, out_activation=nn.ReLU):
    super(Generator_3DDenseNetDecoder, self).__init__()
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

    model = []
    num_layers = 3
    num_input_channels = input_nc
    bn_size = 4
    growth_rate = 16

    model += [
      Dense_3DBlock(num_layers, num_input_channels, True, bn_size, growth_rate, norm_layer, activation, use_bias)
    ]
    num_input_channels = num_layers*growth_rate

    # #upsample
    max_up = int(np.max(np.log2(np.asarray(output_shape) / np.asarray(input_shape))))
    max_down = int(np.log2(input_nc/output_nc))
    assert max_up < max_down, 'upsampling overstep border'

    for i in range(max_up):
      model += [
        nn.Upsample(scale_factor=2, mode=upsample_mode),
        norm_layer(num_input_channels),
        activation,
        nn.Conv3d(num_input_channels, num_input_channels, kernel_size=3, padding=1, bias=use_bias)
      ]

      model += [
        Dense_3DBlock(num_layers, num_input_channels, True, bn_size, growth_rate, norm_layer, activation, use_bias)
      ]
      num_input_channels = growth_rate*num_layers

    # channel reduction
    model += [
      nn.Conv3d(num_input_channels, output_nc, kernel_size=3, padding=1, bias=use_bias)
    ]

    # out activation
    model += [
      out_activation()
    ]

    self.model = nn.Sequential(*model)

  def forward(self, input):
    # #upsample and convolution
    return self.model(input)