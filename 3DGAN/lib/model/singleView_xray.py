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
from lib.model.base_model import Base_Model
from .nets.singleXray3DVolumes import Xray3DVolumes_2DModel
import lib.utils.metrics as Metrics
from .nets.utils import init_net

class CTEncoderDecoder(Base_Model):
  def __init__(self):
    super(CTEncoderDecoder, self).__init__()

  @property
  def name(self):
    return 'singleView_xray2D'

  '''
  Init network architecture
  '''
  def init_network(self, opt):
    Base_Model.init_network(self, opt)

    self.if_pool = opt.if_pool
    self.multi_view = opt.multi_view
    assert len(self.multi_view) > 0

    self.loss_names = ['MSE']
    self.metrics_names = ['Mse', 'CosineSimilarity', 'PSNR']
    self.visual_names = ['G_real', 'G_fake', 'G_input', 'G_Map_fake_F', 'G_Map_real_F', 'G_Map_fake_S', 'G_Map_real_S']

    self.model_names = ['G']

    self.netG = Xray3DVolumes_2DModel()

    self.netG = init_net(self.netG, opt.init_type, self.gpu_ids)


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

    # feature metric loss
    self.criterion = torch.nn.MSELoss()

    self.optimizer_G = torch.optim.Adam(self.netG.parameters(),
                                        lr=opt.lr, betas=(opt.beta1, opt.beta2), weight_decay=1e-4)

    self.optimizers = []
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
    # standard CT dimension
    return value.permute(*tuple(np.argsort(order)))

  def forward(self):
    '''
    self.G_fake is generated object
    self.G_real is GT object
    '''
    # G_fake_D is [B D H W]
    self.G_fake_D1 = self.netG(self.G_input)
    self.G_fake = self.G_fake_D1.permute(0,2,1,3)
    # post processing, used only in testing
    self.post_process(['G_fake'])
    # visualization of x-ray projection
    self.projection_visual()
    # metrics
    self.metrics_evaluation()

  def optimize_parameters(self):
    # forward
    self()
    self.optimizer_G.zero_grad()
    self.loss_MSE = self.criterion(self.G_fake, self.G_real)
    self.loss_MSE.backward()
    self.optimizer_G.step()



