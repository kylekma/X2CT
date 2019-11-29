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
from .loss.gan_loss_package import GANLoss, WGANLoss
from .loss.map_loss import Map_loss
from lib.utils.image_pool import ImagePool
import lib.utils.metrics as Metrics


class TwoD_GD_GAN(Base_Model):
  def __init__(self):
    super(TwoD_GD_GAN, self).__init__()

  @property
  def name(self):
    return 'Noise_3D_G_3D_D_GAN'

  '''
  Init network architecture
  '''
  def init_network(self, opt):
    Base_Model.init_network(self, opt)

    self.loss_names = ['D', 'G']
    self.metrics_names = ['Mse', 'CosineSimilarity']
    self.visual_names = ['G_real', 'G_fake', 'G_input', 'G_map_fake', 'G_map_real']

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
                                   opt.encoder_input_shape, opt.encoder_input_nc, opt.encoder_norm)

    if self.training:
      if opt.ganloss == 'gan':
        use_sigmoid = True
      elif opt.ganloss == 'lsgan':
        use_sigmoid = False
      elif opt.ganloss == 'wgan':
        self.loss_names += ['wasserstein']
        use_sigmoid = False
      elif opt.ganloss == 'wgan_gp':
        self.loss_names += ['wasserstein', 'grad_penalty']
        use_sigmoid = False
      else:
        raise ValueError()

      self.netD = factory.define_D(opt.input_nc_D, opt.ndf,
                                   opt.which_model_netD,
                                   opt.n_layers_D, opt.norm_D,
                                   use_sigmoid, opt.init_type, self.gpu_ids,
                                   opt.discriminator_feature)

      self.fake_pool = ImagePool(opt.pool_size)

  # correspond to visual_names
  def get_normalization_list(self):
    return [
      [self.opt.CT_MEAN_STD[0], self.opt.CT_MEAN_STD[1]],
      [self.opt.CT_MEAN_STD[0], self.opt.CT_MEAN_STD[1]],
      [self.opt.XRAY1_MEAN_STD[0], self.opt.XRAY1_MEAN_STD[1]],
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
    elif opt.ganloss == 'wgan':
      self.criterionGAN = WGANLoss(grad_penalty=False).to(self.device)
    elif opt.ganloss == 'wgan_gp':
      self.criterionGAN = WGANLoss(grad_penalty=True).to(self.device)
    else:
      raise ValueError()

    # identity loss
    if opt.restruction_loss == 'mse':
      print('Restruction loss: MSE')
      self.criterionIdt = torch.nn.MSELoss()
    elif opt.restruction_loss == 'l1':
      print('Restruction loss: l1')
      self.criterionIdt = torch.nn.L1Loss()
    else:
      raise ValueError()

    # feature metric loss
    self.criterionFea = torch.nn.L1Loss()

    # map loss
    self.criterionMap = Map_loss(direct_mean=opt.map_m_type,
                                   predict_transition=self.opt.CT_MIN_MAX,
                                   gt_transition=self.opt.XRAY1_MIN_MAX).to(self.device)

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
    # generate noise
    self.noise = torch.randn(self.opt.batch_size, self.opt.noise_len)

  def calculate_gradient_penalty(self, netD, real_data, fake_data, LAMBDA):
    assert real_data.size(0) == fake_data.size(0), 'Batch size is not consistent'

    batch_size = real_data.size(0)
    alpha = torch.rand(batch_size, 1, requires_grad=True).expand_as(real_data).to(real_data)
    interpolates = (1 - alpha) * real_data + alpha * fake_data
    assert interpolates.requires_grad == True, 'input need to be derivable'

    disc_interpolates = netD(interpolates)[-1]

    gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                                    grad_outputs=torch.ones_like(disc_interpolates),
                                    create_graph=True, retain_graph=True, only_inputs=True,
                                    allow_unused=False)

    gradients = gradients[0].view(gradients[0].size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * LAMBDA

    return gradient_penalty

  # map function
  def output_map(self, v, dim):
    return torch.mean(v, dim=dim, keepdim=True)

  def transition(self, predict):
    new_predict = (predict * (self.opt.CT_MIN_MAX[1]-self.opt.CT_MIN_MAX[0])
                   + self.opt.CT_MIN_MAX[0] -self.opt.XRAY1_MIN_MAX[0])\
                  / (self.opt.XRAY1_MIN_MAX[1]-self.opt.XRAY1_MIN_MAX[0])
    return new_predict

  def forward(self):
    # output is [B 1 D H W]
    self.G_fake_D = self.netG(self.noise)
    # visual object should be [B D H W]
    self.G_fake = torch.squeeze(self.G_fake_D, 1)
    # input of Discriminator is [B 1 D H W]
    self.G_real_D = torch.unsqueeze(self.G_real, 1)
    # value should clip to 0-1 when inference
    if not self.training:
      self.G_fake = torch.clamp(self.G_fake, 0, 1)
    # map
    self.G_map_real = self.transition(self.output_map(self.G_real, 1))
    if self.G_fake_D.dim() == 4:
      self.G_map_fake = self.transition(self.output_map(self.G_fake_D, 1))
    elif self.G_fake_D.dim() == 5:
      self.G_map_fake = self.output_map(self.G_fake_D, 2)
      self.G_map_fake = self.transition(self.G_map_fake.squeeze_(1))
    else:
      raise ValueError
    # metrics
    self.metrics_Mse = Metrics.Mean_Squared_Error(self.G_fake, self.G_real)
    self.metrics_CosineSimilarity = Metrics.Cosine_Similarity(self.G_fake, self.G_real)

  def backward_D(self):
    if self.opt.ganloss == 'gan' or self.opt.ganloss == 'lsgan':

      D_real_pred = self.netD(self.G_real_D)
      self.loss_D_real = self.criterionGAN(D_real_pred, True)

      g_fake_pool = self.fake_pool.query(self.G_fake_D)
      D_fake_pool_pred = self.netD(g_fake_pool.detach())
      self.loss_D_fake = self.criterionGAN(D_fake_pool_pred, False)

      self.loss_D = self.loss_D_fake + self.loss_D_real
      self.loss_grad_penalty = torch.tensor(0.).to(self.loss_D)

      self.loss_D.backward()

    elif self.opt.ganloss == 'wgan':

      D_real_pred = self.netD(self.G_real_D)
      g_fake_pool = self.fake_pool.query(self.G_fake_D)
      D_fake_pool_pred = self.netD(g_fake_pool.detach())

      self.loss_D = torch.mean(D_fake_pool_pred[-1]) - torch.mean(D_real_pred[-1])

      self.loss_grad_penalty = torch.tensor(0.).to(self.loss_D)

      self.loss_D.backward()

    elif self.opt.ganloss == 'wgan_gp':

      D_real_pred = self.netD(self.G_real_D)
      g_fake_pool = self.fake_pool.query(self.G_fake_D)
      D_fake_pool_pred = self.netD(g_fake_pool.detach())

      self.loss_D = torch.mean(D_fake_pool_pred[-1]) - torch.mean(D_real_pred[-1])

      self.loss_grad_penalty = self.calculate_gradient_penalty(self.netD, self.G_real_D, g_fake_pool.detach(), self.opt.wgan_gp_lambda)

      self.loss_D += self.loss_grad_penalty

      self.loss_D.backward(retain_graph=True)

    else:
      raise ValueError()

  def backward_G(self):
    idt_lambda = self.opt.idt_lambda
    fea_m_lambda = self.opt.fea_m_lambda
    map_m_lambda = self.opt.map_m_lambda

    D_real_pred = self.netD(self.G_real_D)

    # Gan loss
    if self.opt.ganloss == 'gan' or self.opt.ganloss == 'lsgan':
      D_fake_pred = self.netD(self.G_fake_D)
      self.loss_G = self.criterionGAN(D_fake_pred, True)
      self.loss_wasserstein = torch.tensor(0.).to(self.loss_G)
    elif self.opt.ganloss == 'wgan' :
      D_fake_pred = self.netD(self.G_fake_D)
      self.loss_G = - torch.mean(D_fake_pred[-1])
      self.loss_wasserstein = torch.mean(D_real_pred[-1]) - torch.mean(D_fake_pred[-1])
    elif self.opt.ganloss == 'wgan_gp':
      D_fake_pred = self.netD(self.G_fake_D)
      self.loss_G = - torch.mean(D_fake_pred[-1])
      self.loss_wasserstein = torch.mean(D_real_pred[-1]) - torch.mean(D_fake_pred[-1])
    else:
      raise ValueError()

    # identity loss
    if idt_lambda != 0:
      self.loss_idt = self.criterionIdt(self.G_fake_D, self.G_real_D) * idt_lambda
    else:
      pass

    # feature metric loss
    if fea_m_lambda != 0:
      loss_G_fea = 0
      feat_weights = 4.0 / (self.opt.n_layers_D + 1)
      # D_weights = 1.0 / self.opt.num_D
      for i in range(len(D_fake_pred)-1):
        loss_G_fea += feat_weights * self.criterionFea(
          D_fake_pred[i], D_real_pred[i].detach()
        ) * fea_m_lambda
      self.loss_fea_m = loss_G_fea
    else:
      pass

    # map loss
    if map_m_lambda > 0:
      self.loss_map_m = self.criterionMap(self.G_map_fake, self.G_input) * map_m_lambda
    else:
      pass

    self.loss_total_G = self.loss_G
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

    if self.opt.ganloss == 'wgan':
      self.clip_weight(self.netD.parameters())

  def optimize_D(self):
    # forward
    self()
    self.set_requires_grad([self.netD], True)
    self.optimizer_D.zero_grad()
    self.backward_D()
    self.optimizer_D.step()

    if self.opt.ganloss == 'wgan':
      self.clip_weight(self.netD.parameters())

  def clip_weight(self, parameters, clip_bounds=(-0.01, 0.01)):
    # weight clip
    for para in parameters:
      para.data.clamp_(min=clip_bounds[0], max=clip_bounds[1])
