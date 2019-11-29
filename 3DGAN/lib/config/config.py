# ------------------------------------------------------------------------------
# Copyright (c) Tencent
# Licensed under the GPLv3 License.
# Created by Kai Ma (makai0324@gmail.com)
# ------------------------------------------------------------------------------

from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

from easydict import EasyDict
import os
import numpy as np

__C = EasyDict()
cfg = __C

# Model Path
__C.MODEL_SAVE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'save_models'))
__C.CT_MIN_MAX = [0, 5800]
__C.XRAY1_MIN_MAX = [0, 1200]
__C.XRAY2_MIN_MAX = [0, 1700]
__C.CT_MEAN_STD = [0., 1.0]
__C.XRAY1_MEAN_STD = [0., 1.0]
__C.XRAY2_MEAN_STD = [0., 1.0]

'''
Network
  Generator
'''
__C.NETWORK = EasyDict()
# of input image channels
__C.NETWORK.input_nc_G = 3
# of output image channels
__C.NETWORK.output_nc_G = 3
# of gen filters in first conv layer
__C.NETWORK.ngf = 64
# selects model to use for netG
__C.NETWORK.which_model_netG = 'resnet_generator'
# instance normalization or batch normalization
__C.NETWORK.norm_G = 'instance'
# no dropout for the generator
__C.NETWORK.no_dropout = False
# network initialization [normal|xavier|kaiming|orthogonal]
__C.NETWORK.init_type = 'normal'
# gan, lsgan, wgan, wgan_gp
__C.NETWORK.ganloss = 'lsgan'
# down sampling
__C.NETWORK.n_downsampling = 3
__C.NETWORK.n_blocks = 9
# activation
__C.NETWORK.activation_type = 'relu'

'''
Network
  Discriminator
'''
# of input image channels
__C.NETWORK.input_nc_D = 3
# of output image channels
# __C.NETWORK.output_nc_D = 1
# of discrim filters in first conv layer
__C.NETWORK.ndf = 64
# selects model to use for netD
__C.NETWORK.which_model_netD = 'basic'
# only used if which_model_netD==n_layers, dtype = int
__C.NETWORK.n_layers_D = 3
# instance normalization or batch normalization, dtype = str
__C.NETWORK.norm_D = 'instance3d'
# output channels of discriminator network, dtype = int
__C.NETWORK.n_out_ChannelsD = 1
__C.NETWORK.pool_size = 50
__C.NETWORK.if_pool = False
__C.NETWORK.num_D = 3
# add condition to discriminator network
__C.NETWORK.conditional_D = False

# of input image channels
__C.NETWORK.map_input_nc_D = 1
# of discrim filters in first conv layer
__C.NETWORK.map_ndf = 64
# selects model to use for netD
__C.NETWORK.map_which_model_netD = 'basic'
# only used if which_model_netD==n_layers
__C.NETWORK.map_n_layers_D = 3
# instance normalization or batch normalization, dtype = str
__C.NETWORK.map_norm_D = 'instance'
# output channels of discriminator network, dtype = int
__C.NETWORK.map_n_out_ChannelsD = 1
__C.NETWORK.map_pool_size = 50
__C.NETWORK.map_num_D = 3

'''
Train
'''
__C.TRAIN = EasyDict()
# initial learning rate for adam
__C.TRAIN.lr = 0.0002
# momentum term of adam
__C.TRAIN.beta1 = 0.5
__C.TRAIN.beta2 = 0.9
# if true, takes images in order to make batches, otherwise takes them randomly
__C.TRAIN.serial_batches = False
__C.TRAIN.batch_size = 1
# threads for loading data
__C.TRAIN.nThreads = 5
# __C.TRAIN.max_epoch = 10
# learning rate policy: lambda|step|plateau
__C.TRAIN.lr_policy = 'lambda'
# of iter at starting learning rate
__C.TRAIN.niter = 100
# of iter to linearly decay learning rate to zero
__C.TRAIN.niter_decay = 100
# multiply by a gamma every lr_decay_iters iterations
__C.TRAIN.lr_decay_iters = 50
# frequency of showing training results on console
__C.TRAIN.print_freq = 10
# frequency of showing training results on console
__C.TRAIN.print_img_freq = 200
# save model
__C.TRAIN.save_latest_freq = 3000
# save model frequent
__C.TRAIN.save_epoch_freq = 5
__C.TRAIN.begin_save_epoch = 0

__C.TRAIN.weight_decay_if = False

'''
TEST
'''
__C.TEST = EasyDict()
__C.TEST.howmany_in_train = 10

'''
Data
Augmentation
'''
__C.DATA_AUG = EasyDict()
__C.DATA_AUG.select_slice_num = 0
__C.DATA_AUG.fine_size = 256
__C.DATA_AUG.ct_channel = 256
__C.DATA_AUG.xray_channel = 1
__C.DATA_AUG.resize_size = 289

'''
2D GAN define loss
'''
__C.TD_GAN = EasyDict()
# identity loss
__C.TD_GAN.idt_lambda = 10.
__C.TD_GAN.idt_reduction = 'elementwise_mean'
__C.TD_GAN.idt_weight = 0.
__C.TD_GAN.idt_weight_range = [0., 1.]
__C.TD_GAN.restruction_loss = 'l1'
# perceptual loss
__C.TD_GAN.fea_m_lambda = 10.
# output of discriminator
__C.TD_GAN.discriminator_feature = True
# wgan-gp
__C.TD_GAN.wgan_gp_lambda = 10.
# identity loss of map
__C.TD_GAN.map_m_lambda = 0.
# 'l1' or 'mse'
__C.TD_GAN.map_m_type = 'l1'
__C.TD_GAN.fea_m_map_lambda = 10.
# Discriminator train times
__C.TD_GAN.critic_times = 1

'''
3D GD-GAN define structure
'''
__C.D3_GAN = EasyDict()
__C.D3_GAN.noise_len = 1000
__C.D3_GAN.input_shape = [4,4,4]
# __C.D3_GAN.input_shape_nc = 512
__C.D3_GAN.output_shape = [128,128,128]
# __C.D3_GAN.output_shape_nc = 1
__C.D3_GAN.encoder_input_shape = [128, 128]
__C.D3_GAN.encoder_input_nc = 1
__C.D3_GAN.encoder_norm = 'instance'
__C.D3_GAN.encoder_blocks = 4
__C.D3_GAN.multi_view = [1,2,3]
__C.D3_GAN.min_max_norm = False
__C.D3_GAN.skip_number = 1
# DoubleBlockLinearUnit Activation [low high k]
__C.D3_GAN.dblu = [0., 1.0, 1.0]

'''
CT GAN
'''
__C.CTGAN = EasyDict()
# input x-ray direction, 'H'-FrontBehind 'D'-UpDown 'W'-LeftRight
# 'HDW' Means that deepness is 'H' and projection in plane of 'DW'
#  relative to CT.
__C.CTGAN.Xray1_Direction = 'HDW'
__C.CTGAN.Xray2_Direction = 'WDH'
# dimension order of input CT is 'DHW'(should add 'NC'-01 to front when training)
__C.CTGAN.CTOrder = [0, 1, 2, 3, 4]
# NCHDW to xray1 and NCWDH to xray2
__C.CTGAN.CTOrder_Xray1 = [0, 1, 3, 2, 4]
__C.CTGAN.CTOrder_Xray2 = [0, 1, 4, 2, 3]
# identity loss'weight
__C.CTGAN.idt_lambda = 1.0
__C.CTGAN.idt_reduction = 'elementwise_mean'
__C.CTGAN.idt_weight = 0.
__C.CTGAN.idt_weight_range = [0., 1.]
# 'l1' or 'mse'
__C.CTGAN.idt_loss = 'l1'
# feature metrics loss
__C.CTGAN.feature_D_lambda = 0.
# projection loss'weight
__C.CTGAN.map_projection_lambda = 0.
# 'l1' or 'mse'
__C.CTGAN.map_projection_loss = 'l1'
# gan loss'weight
__C.CTGAN.gan_lambda = 1.0
# multiView GAN auxiliary loss
__C.CTGAN.auxiliary_lambda = 0.
# 'l1' or 'mse'
__C.CTGAN.auxiliary_loss = 'mse'
# map discriminator
__C.CTGAN.feature_D_map_lambda = 0.
__C.CTGAN.map_gan_lambda = 1.0

def cfg_from_yaml(filename):
  '''
  Load a config file and merge it into the default options
  :param filename:
  :return:
  '''
  import yaml
  with open(filename, 'r') as f:
    yaml_cfg = EasyDict(yaml.load(f))
  _merge_a_into_b(yaml_cfg, __C)

def print_easy_dict(easy_dict):
  print('==='*10)
  print('====YAML Parameters')
  for k,v in easy_dict.__dict__.items():
    print('{}: {}'.format(k, v))
  print('==='*10)

def merge_dict_and_yaml(in_dict, easy_dict):
  if type(easy_dict) is not EasyDict:
    return in_dict
  easy_list = _easy_dict_squeeze(easy_dict)
  for (k, v) in easy_list:
    if k in in_dict:
      raise KeyError('The same Key appear {}/{}'.format(k,k))
  out_dict = EasyDict(dict(easy_list + list(in_dict.items())))
  return out_dict

def _easy_dict_squeeze(easy_dict):
  if type(easy_dict) is not EasyDict:
    print('Not EasyDict!!!')
    return []

  total_list = []
  for k, v in easy_dict.items():
    # recursively merge dicts
    if type(v) is EasyDict:
      try:
        total_list += _easy_dict_squeeze(v)
      except:
        print('Error under config key: {}'.format(k))
        raise
    else:
      total_list.append((k, v))
  return total_list

def _merge_a_into_b(a, b):
  '''
  Merge easyDict a to easyDict b
  :param a: from easyDict
  :param b: to easyDict
  :return:
  '''
  if type(a) is not EasyDict:
    return

  for k, v in a.items():
    # check k in a or not
    if k not in b:
      raise KeyError('{} is not a valid config key'.format(k))

    old_type = type(b[k])
    if old_type is not type(v):
      if isinstance(b[k], np.ndarray):
        v = np.array(v, dtype=b[k].type)
      else:
        raise ValueError('Type mismatch ({} vs. {})'
                         'for config key: {}'.format(type(b[k]), type(v), k))
    # recursively merge dicts
    if type(v) is EasyDict:
      try:
        _merge_a_into_b(a[k], b[k])
      except:
        print('Error under config key: {}'.format(k))
        raise
    else:
      b[k] = v







