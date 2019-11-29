# ------------------------------------------------------------------------------
# Copyright (c) Tencent
# Licensed under the GPLv3 License.
# Created by Kai Ma (makai0324@gmail.com)
# ------------------------------------------------------------------------------

from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import torch
from lib.model.base_model import Base_Model
import lib.model.nets.factory as factory
from .loss.multi_gan_loss import RestructionLoss
import lib.utils.metrics as Metrics
import numpy as np

class CTGAN(Base_Model):
  def __init__(self):
    super(CTGAN, self).__init__()

  @property
  def name(self):
    return 'singleView_ED3D'

  '''
  Init network architecture
  '''
  def init_network(self, opt):
    Base_Model.init_network(self, opt)

    self.if_pool = opt.if_pool
    self.multi_view = opt.multi_view
    assert len(self.multi_view) > 0

    self.metrics_names = ['Mse', 'CosineSimilarity', 'PSNR']
    self.visual_names = ['G_real', 'G_fake', 'G_input', 'G_Map_fake_F', 'G_Map_real_F', 'G_Map_fake_S', 'G_Map_real_S']

    self.netG = factory.define_3DG(opt.noise_len, opt.input_shape, opt.output_shape,
                                   opt.input_nc_G, opt.output_nc_G, opt.ngf, opt.which_model_netG,
                                   opt.n_downsampling, opt.norm_G, not opt.no_dropout,
                                   opt.init_type, self.gpu_ids, opt.n_blocks,
                                   opt.encoder_input_shape, opt.encoder_input_nc, opt.encoder_norm,
                                   opt.encoder_blocks, opt.skip_number, opt.activation_type, opt=opt)

    self.loss_names = ['idt']
    self.model_names = ['G']

    # map loss
    if self.opt.map_projection_lambda > 0:
      self.loss_names += ['map_m']

  # correspond to visual_names
  def get_normalization_list(self):
    return [
      [self.opt.CT_MEAN_STD[0], self.opt.CT_MEAN_STD[1]],
      [self.opt.CT_MEAN_STD[0], self.opt.CT_MEAN_STD[1]],
      [self.opt.XRAY1_MEAN_STD[0], self.opt.XRAY1_MEAN_STD[1]],
      [self.opt.CT_MEAN_STD[0], self.opt.CT_MEAN_STD[1]],
      [self.opt.CT_MEAN_STD[0], self.opt.CT_MEAN_STD[1]],
      [self.opt.CT_MEAN_STD[0], self.opt.CT_MEAN_STD[1]],
      [self.opt.CT_MEAN_STD[0], self.opt.CT_MEAN_STD[1]]
    ]

  def init_loss(self, opt):
    Base_Model.init_loss(self, opt)

    # #####################
    # define loss functions
    # #####################

    # identity loss
    self.criterionIdt = RestructionLoss(opt.idt_loss, opt.idt_reduction).to(self.device)

    # map loss
    self.criterionMap = RestructionLoss(opt.map_projection_loss).to(self.device)

    # #####################
    # initialize optimizers
    # #####################
    self.optimizers = []
    if self.opt.weight_decay_if:
      self.optimizer_G = torch.optim.Adam(self.netG.parameters(),
                                          lr=opt.lr, betas=(opt.beta1, opt.beta2), weight_decay=1e-4)
    else:
      self.optimizer_G = torch.optim.Adam(self.netG.parameters(),
                                          lr=opt.lr, betas=(opt.beta1, opt.beta2))
    self.optimizers.append(self.optimizer_G)

  '''
    Train -Forward and Backward
  '''
  def set_input(self, input):
    self.G_input = input[1].to(self.device)
    self.G_real = input[0].to(self.device)
    self.image_paths = input[2:]

  # map function
  def output_map(self, v, dim):
    '''
    :param v: tensor
    :param dim:  dimension be reduced
    :return:
      N1HW
    '''
    ori_dim = v.dim()
    # tensor [NDHW]
    if ori_dim == 4:
      map = torch.mean(torch.abs(v), dim=dim)
      # [NHW] => [NCHW]
      return map.unsqueeze(1)
    # tensor [NCDHW] and c==1
    elif ori_dim == 5:
      # [NCHW]
      map = torch.mean(torch.abs(v), dim=dim)
      return map
    else:
      raise NotImplementedError()

  def transition(self, predict):
    p_max, p_min = predict.max(), predict.min()
    new_predict = (predict - p_min) / (p_max - p_min)
    return new_predict

  def ct_unGaussian(self, value):
    return value * self.opt.CT_MEAN_STD[1] + self.opt.CT_MEAN_STD[0]

  def ct_Gaussian(self, value):
    return (value - self.opt.CT_MEAN_STD[0]) / self.opt.CT_MEAN_STD[1]

  def post_process(self, attributes_name):
    if not self.training:
      if self.opt.CT_MEAN_STD[0] == 0 and self.opt.CT_MEAN_STD[0] == 0:
        for name in attributes_name:
          setattr(self, name, torch.clamp(getattr(self, name), 0, 1))
      elif self.opt.CT_MEAN_STD[0] == 0.5 and self.opt.CT_MEAN_STD[0] == 0.5:
        for name in attributes_name:
          setattr(self, name, torch.clamp(getattr(self, name), -1, 1))
      else:
        raise NotImplementedError()

  def projection_visual(self):
    # map F is projected in dimension of H
    self.G_Map_real_F = self.transition(self.output_map(self.ct_unGaussian(self.G_real), 2))
    self.G_Map_fake_F = self.transition(self.output_map(self.ct_unGaussian(self.G_fake), 2))
    # map S is projected in dimension of W
    self.G_Map_real_S = self.transition(self.output_map(self.ct_unGaussian(self.G_real), 3))
    self.G_Map_fake_S = self.transition(self.output_map(self.ct_unGaussian(self.G_fake), 3))

  def metrics_evaluation(self):
    # 3D metrics including mse, cs and psnr
    g_fake_unNorm = self.ct_unGaussian(self.G_fake)
    g_real_unNorm = self.ct_unGaussian(self.G_real)

    self.metrics_Mse = Metrics.Mean_Squared_Error(g_fake_unNorm, g_real_unNorm)
    self.metrics_CosineSimilarity = Metrics.Cosine_Similarity(g_fake_unNorm, g_real_unNorm)
    self.metrics_PSNR = Metrics.Peak_Signal_to_Noise_Rate(g_fake_unNorm, g_real_unNorm, PIXEL_MAX=1.0)

  def dimension_order_std(self, value, order):
    return value.permute(*tuple(np.argsort(order)))

  def forward(self):
    '''
      self.G_fake is generated object
      self.G_real is GT object
      '''
    # G_fake_D is [B 1 D H W]
    self.G_fake_D1 = self.netG(self.G_input)
    self.G_fake_D = self.dimension_order_std(self.G_fake_D1, self.opt.CTOrder_Xray1)
    # visual object should be [B D H W]
    self.G_fake = torch.squeeze(self.G_fake_D, 1)
    # input of Discriminator is [B 1 D H W]
    self.G_real_D = torch.unsqueeze(self.G_real, 1)
    # post processing, used only in testing
    self.post_process(['G_fake'])
    if not self.training:
      # visualization of x-ray projection
      self.projection_visual()
      # metrics
      self.metrics_evaluation()
    # multi-view projection maps for training
    # Note: self.G_real_D and self.G_fake_D are in dimension order of 'NCDHW'
    if self.training:
      for i in self.multi_view:
        out_map = self.output_map(self.ct_unGaussian(self.G_real_D), i + 1)
        out_map = self.ct_Gaussian(out_map)
        setattr(self, 'G_Map_{}_real'.format(i), out_map)

        out_map = self.output_map(self.ct_unGaussian(self.G_fake_D), i + 1)
        out_map = self.ct_Gaussian(out_map)
        setattr(self, 'G_Map_{}_fake'.format(i), out_map)

  def optimize_parameters(self):
    self()
    self.optimizer_G.zero_grad()

    total_loss = 0
    idt_lambda = self.opt.idt_lambda
    map_projection_lambda = self.opt.map_projection_lambda

    # focus area weight assignment
    if self.opt.idt_reduction == 'none' and self.opt.idt_weight > 0:
      idt_low, idt_high = self.opt.idt_weight_range
      idt_weight = self.opt.idt_weight
      loss_idt = self.criterionIdt(self.G_fake_D, self.G_real_D)
      mask = (self.G_real_D > idt_low) & (self.G_real_D < idt_high)
      loss_idt[mask] = loss_idt[mask] * idt_weight
      self.loss_idt = loss_idt.mean() * idt_lambda
    else:
      self.loss_idt = self.criterionIdt(self.G_fake_D, self.G_real_D) * idt_lambda
    total_loss += self.loss_idt

    if self.opt.map_projection_lambda > 0:
      self.loss_map_m = 0.
      for direction in self.multi_view:
        self.loss_map_m += self.criterionMap(
          getattr(self, 'G_Map_{}_fake'.format(direction)),
          getattr(self, 'G_Map_{}_real'.format(direction))) * map_projection_lambda
      self.loss_map_m = self.loss_map_m / len(self.multi_view)
      total_loss += self.loss_map_m
    total_loss.backward()
    self.optimizer_G.step()