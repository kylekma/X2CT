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

# Defines the GAN loss which uses either LSGAN or the regular GAN.
# When LSGAN is used, it is basically same as MSELoss,
# but it abstracts away the need to create the target label tensor
# that has the same size as the input
'''
GANLoss
input: 
  [ ... V(probability Or distance) ] , V.size() = (BCHW)
'''
class GANLoss(nn.Module):
  def __init__(self, use_lsgan=True, target_real_label=1.0, target_fake_label=0.0):
    super(GANLoss, self).__init__()
    self.register_buffer('real_label', torch.tensor(target_real_label))
    self.register_buffer('fake_label', torch.tensor(target_fake_label))
    self.use_lsgan = use_lsgan
    if use_lsgan:
      self.loss = nn.MSELoss()
      print('GAN loss: {}'.format('LSGAN'))
    else:
      self.loss = nn.BCELoss()
      print('GAN loss: {}'.format('Normal'))

  def get_target_tensor(self, input, target_is_real):
    if target_is_real:
      target_tensor = self.real_label
    else:
      target_tensor = self.fake_label
    return target_tensor.expand_as(input)

  def forward(self, input, target_is_real):
    input_in = input[-1]
    target_tensor = self.get_target_tensor(input_in, target_is_real)
    return self.loss(input_in, target_tensor)


class WGANLoss(nn.Module):
  def __init__(self, grad_penalty=False, clip_bounds=(-0.01, 0.01)):
    super(WGANLoss, self).__init__()
    self.grad_penalty = grad_penalty
    if grad_penalty:
      print('GAN loss: {}'.format('WGAN-GP'))
    else:
      self.clip_bounds = clip_bounds
      print('GAN loss: {}'.format('WGAN'))

  def get_mean(self, input):
    input_mean = torch.mean(input)
    return input_mean

  def forward(self, input):
    # WGAN
    if not self.grad_penalty:
      disc_fake, disc_real, parameters = input
      disc_fake, disc_real = disc_fake[-1], disc_real[-1]
      gen_cost = -self.get_mean(disc_fake)
      disc_cost = self.get_mean(disc_fake) - self.get_mean(disc_real)
      wasserstein = -disc_cost
      # weight clip
      for para in parameters:
        para.data.clamp_(min=self.clip_bounds[0], max=self.clip_bounds[1])

      return gen_cost, disc_cost, wasserstein
    # WGAN-GP
    else:
      disc_fake, disc_real, gradient_penalty = input
      disc_fake, disc_real = disc_fake[-1], disc_real[-1]
      gen_cost = -self.get_mean(disc_fake)
      disc_cost = self.get_mean(disc_fake) - self.get_mean(disc_real)
      wasserstein = -disc_cost
      disc_cost += gradient_penalty

      return gen_cost, disc_cost, wasserstein

# Restruction Loss
class RestructionLoss(nn.Module):

  def __init__(self, distance='l1'):
    super(RestructionLoss, self).__init__()
    if distance == 'l1':
      self.loss = nn.L1Loss()
    elif distance == 'mse':
      self.loss = nn.MSELoss()
    else:
      raise NotImplementedError()

  def forward(self, gt, pred):
    return self.loss(gt, pred)





