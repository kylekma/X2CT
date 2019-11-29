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
from .loss.multi_gan_loss import GANLoss, RestructionLoss
from lib.utils.image_pool import ImagePool
import lib.utils.metrics as Metrics

class CTGAN(Base_Model):
  def __init__(self):
    super(CTGAN, self).__init__()

  @property
  def name(self):
    return 'singleView_GAN2DD'

  '''
  Init network architecture
  '''
  def init_network(self, opt):
    Base_Model.init_network(self, opt)

    self.if_pool = opt.if_pool
    self.multi_view = opt.multi_view
    self.conditional_D = opt.conditional_D
    assert len(self.multi_view) > 0

    self.loss_names = ['D', 'G']
    self.metrics_names = ['Mse', 'CosineSimilarity', 'PSNR']
    self.visual_names = ['G_real', 'G_fake', 'G_input', 'G_Map_fake_F', 'G_Map_real_F', 'G_Map_fake_S', 'G_Map_real_S']

    # identity loss
    if self.opt.idt_lambda > 0:
      self.loss_names += ['idt']

    # feature metric loss
    if self.opt.fea_m_lambda > 0:
      self.loss_names += ['fea_m']

    # map loss
    if self.opt.map_m_lambda > 0:
      self.loss_names += ['map_m']

    if self.training:
      self.model_names = ['G', 'D']
    else:  # during test time, only load Gs
      self.model_names = ['G']

    self.netG = factory.define_3DG(opt.noise_len, opt.input_shape, opt.output_shape,
                                   opt.input_nc_G, opt.output_nc_G, opt.ngf, opt.which_model_netG,
                                   opt.n_downsampling, opt.norm_G, not opt.no_dropout,
                                   opt.init_type, self.gpu_ids, opt.n_blocks,
                                   opt.encoder_input_shape, opt.encoder_input_nc, opt.encoder_norm,
                                   opt.encoder_blocks, opt.skip_number, opt.activation_type, opt=opt)

    if self.training:
      if opt.ganloss == 'gan':
        use_sigmoid = True
      elif opt.ganloss == 'lsgan':
        use_sigmoid = False
      else:
        raise ValueError()

      # conditional Discriminator
      if self.conditional_D:
        d_input_channels = opt.input_nc_D + 1
      else:
        d_input_channels = opt.input_nc_D
      self.netD = factory.define_D(d_input_channels, opt.ndf,
                                   opt.which_model_netD,
                                   opt.n_layers_D, opt.norm_D,
                                   use_sigmoid, opt.init_type, self.gpu_ids,
                                   opt.discriminator_feature, num_D=opt.num_D, n_out_channels=opt.n_out_ChannelsD)
      if self.if_pool:
        self.fake_pool = ImagePool(opt.pool_size)

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
    # GAN loss
    if opt.ganloss == 'gan':
      self.criterionGAN = GANLoss(use_lsgan=False).to(self.device)
    elif opt.ganloss == 'lsgan':
      self.criterionGAN = GANLoss(use_lsgan=True).to(self.device)
    else:
      raise ValueError()

    # identity loss
    self.criterionIdt = RestructionLoss(opt.restruction_loss).to(self.device)

    # feature metric loss
    self.criterionFea = torch.nn.L1Loss()

    # map loss
    self.criterionMap = RestructionLoss(opt.map_m_type).to(self.device)

    # #####################
    # initialize optimizers
    # #####################
    self.optimizer_G = torch.optim.Adam(self.netG.parameters(),
                                        lr=opt.lr, betas=(opt.beta1, opt.beta2))
    self.optimizer_D = torch.optim.Adam(self.netD.parameters(),
                                        lr=opt.lr, betas=(opt.beta1, opt.beta2))
    self.optimizers = []
    self.optimizers.append(self.optimizer_G)
    self.optimizers.append(self.optimizer_D)

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
    :param v:
      n(c=1)dhw
    :param dim:
    :return:
    '''
    ori_dim = v.dim()
    map = torch.mean(v, dim=dim, keepdim=True)
    shape_l = [i for i in map.size()[1:] if i != 1]
    shape_h = [map.size(0)]
    shape_m = [1 for i in range(map.dim() - len(shape_l) - 1)]
    map = map.view(*(shape_h + shape_m + shape_l))
    return map

  def transition(self, predict):
    new_predict = predict * (self.opt.CT_MIN_MAX[1]-self.opt.CT_MIN_MAX[0]) + self.opt.CT_MIN_MAX[0]
    p_max, p_min = new_predict.max(), new_predict.min()
    new_predict = (new_predict - p_min) / (p_max - p_min)
    return new_predict

  def ct_unZeroOne(self, value):
    return value * (self.opt.CT_MIN_MAX[1]-self.opt.CT_MIN_MAX[0]) + self.opt.CT_MIN_MAX[0]

  def xray_unZeroOne(self, value):
    return value * (self.opt.XRAY1_MIN_MAX[1]-self.opt.XRAY1_MIN_MAX[0]) + self.opt.XRAY1_MIN_MAX[0]

  def ct_unGaussian(self, value):
    return value * self.opt.CT_MEAN_STD[1] + self.opt.CT_MEAN_STD[0]

  def ct_Gaussian(self, value):
    return (value - self.opt.CT_MEAN_STD[0]) / self.opt.CT_MEAN_STD[1]

  def xray_unGaussian(self, value):
    return value * self.opt.XRAY1_MEAN_STD[1] + self.opt.XRAY1_MEAN_STD[0]

  def forward(self):
    # output is [B D H W]
    self.G_fake = self.netG(self.G_input)
    # visual object should be [B D H W]
    self.G_fake_D = self.G_fake
    if not self.training:
      if self.opt.CT_MEAN_STD[0] == 0:
        self.G_fake = torch.clamp(self.G_fake, 0, 1)
      elif self.opt.CT_MEAN_STD[0] == 0.5:
        self.G_fake = torch.clamp(self.G_fake, -1, 1)
      else:
        raise NotImplementedError()
    # input of Discriminator is [B D H W]
    self.G_real_D = self.G_real
    if self.conditional_D:
      self.G_condition_D = self.G_input
    # map
    self.G_Map_real_F = self.transition(self.output_map(self.ct_unGaussian(self.G_real_D), 1))
    self.G_Map_fake_F = self.transition(self.output_map(self.ct_unGaussian(self.G_fake_D), 1))
    self.G_Map_real_S = self.transition(self.output_map(self.ct_unGaussian(self.G_real_D), 3))
    self.G_Map_fake_S = self.transition(self.output_map(self.ct_unGaussian(self.G_fake_D), 3))

    if self.training:
      for i in self.multi_view:
        out_map = self.output_map(self.ct_unGaussian(self.G_real_D), i)
        out_map = self.ct_Gaussian(self.transition(out_map))
        setattr(self, 'G_Map_{}_real'.format(i), out_map)

        out_map = self.output_map(self.ct_unGaussian(self.G_fake_D), i)
        out_map = self.ct_Gaussian(self.transition(out_map))
        setattr(self, 'G_Map_{}_fake'.format(i), out_map)

    # metrics
    g_fake_unNorm = self.ct_unGaussian(self.G_fake)
    g_real_unNorm = self.ct_unGaussian(self.G_real)

    self.metrics_Mse = Metrics.Mean_Squared_Error(g_fake_unNorm, g_real_unNorm)
    self.metrics_CosineSimilarity = Metrics.Cosine_Similarity(g_fake_unNorm, g_real_unNorm)
    self.metrics_PSNR = Metrics.Peak_Signal_to_Noise_Rate(g_fake_unNorm, g_real_unNorm, PIXEL_MAX=1.0)

  # feature metrics loss
  def feature_metric_loss(self, D_fake_pred, D_real_pred, loss_weight, num_D, feat_weights, criterionFea):
      fea_m_lambda = loss_weight
      loss_G_fea = 0
      feat_weights = feat_weights
      D_weights = 1.0 / num_D
      weight = feat_weights * D_weights

      # multi-scale discriminator
      if isinstance(D_fake_pred[0], list):
        for di in range(num_D):
          for i in range(len(D_fake_pred[di]) - 1):
            loss_G_fea += weight * criterionFea(D_fake_pred[di][i], D_real_pred[di][i].detach()) * fea_m_lambda
      # single discriminator
      else:
        for i in range(len(D_fake_pred) - 1):
          loss_G_fea += feat_weights * criterionFea(D_fake_pred[i], D_real_pred[i].detach()) * fea_m_lambda

      return loss_G_fea

  def backward_D_basic(self, D_network, input_real, input_fake, fake_pool, criterionGAN):
    if self.opt.ganloss == 'gan' or self.opt.ganloss == 'lsgan':
      D_real_pred = D_network(input_real)
      gan_loss_real = criterionGAN(D_real_pred, True)

      if self.if_pool:
        g_fake_pool = fake_pool.query(input_fake)
      else:
        g_fake_pool = input_fake
      D_fake_pool_pred = D_network(g_fake_pool.detach())
      gan_loss_fake = criterionGAN(D_fake_pool_pred, False)
      gan_loss = gan_loss_fake + gan_loss_real
      gan_loss.backward()
      return gan_loss
    else:
      raise NotImplementedError()

  def backward_D(self):
    if self.opt.ganloss == 'gan' or self.opt.ganloss == 'lsgan':
      if self.conditional_D:
        fake_input = torch.cat([self.G_condition_D, self.G_fake_D], 1)
        real_input = torch.cat([self.G_condition_D, self.G_real_D], 1)
      else:
        fake_input = self.G_fake_D
        real_input = self.G_real_D
      self.loss_D = self.backward_D_basic(self.netD, real_input, fake_input,self.fake_pool if self.if_pool else None, self.criterionGAN)
    else:
      raise ValueError()

  def backward_G_basic(self, D_network, input_real, input_fake, criterionGAN):
    if self.opt.ganloss == 'gan' or self.opt.ganloss == 'lsgan':
      D_fake_pred = D_network(input_fake)
      loss_G = criterionGAN(D_fake_pred, True)
    else:
      raise ValueError()
    return loss_G, D_fake_pred

  def backward_G(self):
    idt_lambda = self.opt.idt_lambda
    fea_m_lambda = self.opt.fea_m_lambda
    map_m_lambda = self.opt.map_m_lambda

    ############################################
    # BackBone GAN
    ############################################
    # GAN loss
    if self.conditional_D:
      fake_input = torch.cat([self.G_condition_D, self.G_fake_D], 1)
      real_input = torch.cat([self.G_condition_D, self.G_real_D], 1)
    else:
      fake_input = self.G_fake_D
      real_input = self.G_real_D

    self.loss_G, D_fake_pred = self.backward_G_basic(self.netD, None, fake_input, self.criterionGAN)

    # identity loss
    if idt_lambda != 0:
      self.loss_idt = self.criterionIdt(self.G_fake_D, self.G_real_D) * idt_lambda

    D_real_pred = self.netD(real_input)

    # feature metric loss
    if fea_m_lambda != 0:
      self.loss_fea_m = self.feature_metric_loss(D_fake_pred, D_real_pred, loss_weight=fea_m_lambda, num_D=self.opt.num_D, feat_weights=4.0 / (self.opt.n_layers_D + 1), criterionFea=self.criterionFea)

    # map loss
    if map_m_lambda > 0:
      self.loss_map_m = 0.
      for direction in self.multi_view:
        self.loss_map_m += self.criterionMap(
          getattr(self, 'G_Map_{}_fake'.format(direction)),
          getattr(self, 'G_Map_{}_real'.format(direction))) * map_m_lambda
      self.loss_map_m = self.loss_map_m / len(self.multi_view)

    # 0.0 must be add to loss_total, otherwise loss_G will
    # be changed when loss_total_G change
    self.loss_total_G = self.loss_G + 0.0
    if idt_lambda > 0:
      self.loss_total_G += self.loss_idt
    if fea_m_lambda > 0:
      self.loss_total_G += self.loss_fea_m
    if map_m_lambda > 0:
      self.loss_total_G += self.loss_map_m

    self.loss_total_G.backward()

  def optimize_parameters(self):
    # forward
    self()
    self.set_requires_grad([self.netD], False)
    self.optimizer_G.zero_grad()
    self.backward_G()
    self.optimizer_G.step()

    self.set_requires_grad([self.netD], True)
    self.optimizer_D.zero_grad()
    self.backward_D()
    self.optimizer_D.step()

  def optimize_D(self):
    # forward
    self()
    self.set_requires_grad([self.netD], True)
    self.optimizer_D.zero_grad()
    self.backward_D()
    self.optimizer_D.step()


