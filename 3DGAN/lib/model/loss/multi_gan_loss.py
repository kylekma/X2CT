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


class GANLoss(nn.Module):
  def __init__(self, use_lsgan=True, target_real_label=1.0, target_fake_label=0.0):
    super(GANLoss, self).__init__()
    self.real_label = target_real_label
    self.fake_label = target_fake_label
    self.real_label_tensor = None
    self.fake_label_tensor = None
    if use_lsgan:
      self.loss = nn.MSELoss()
      print('GAN loss: {}'.format('LSGAN'))
    else:
      self.loss = nn.BCELoss()
      print('GAN loss: {}'.format('Normal'))

  def get_target_tensor(self, input, target_is_real):
    target_tensor = None
    if target_is_real:
      create_label = ((self.real_label_tensor is None) or
                      (self.real_label_tensor.numel() != input.numel()))
      if create_label:
        real_tensor = torch.ones(input.size(), dtype=torch.float).fill_(self.real_label)
        self.real_label_tensor = real_tensor.to(input)
      target_tensor = self.real_label_tensor
    else:
      create_label = ((self.fake_label_tensor is None) or
                      (self.fake_label_tensor.numel() != input.numel()))
      if create_label:
        fake_tensor = torch.ones(input.size(), dtype=torch.float).fill_(self.fake_label)
        self.fake_label_tensor = fake_tensor.to(input)
      target_tensor = self.fake_label_tensor
    return target_tensor

  def forward(self, input, target_is_real):
    # for multi_scale_discriminator
    if isinstance(input[0], list):
      loss = 0
      for input_i in input:
        pred = input_i[-1]
        target_tensor = self.get_target_tensor(pred, target_is_real)
        loss += self.loss(pred, target_tensor)
      return loss
    # for patch_discriminator
    else:
      target_tensor = self.get_target_tensor(input[-1], target_is_real)
      return self.loss(input[-1], target_tensor)


class WGANLoss(nn.Module):
  def __init__(self, grad_penalty=False):
    super(WGANLoss, self).__init__()
    self.grad_penalty = grad_penalty
    if grad_penalty:
      print('GAN loss: {}'.format('WGAN-GP'))
    else:
      print('GAN loss: {}'.format('WGAN'))

  def get_mean(self, input):
    input_mean = torch.mean(input)
    return input_mean

  def forward(self, input_fake, input_real=None, is_G=True):
    if is_G:
      assert input_real is None
    cost = 0.
    # for multi_scale_discriminator
    if isinstance(input_fake[0], list):
      for i in range(len(input_fake)):
        if is_G:
          disc_fake = input_fake[i][-1]
          cost += (-self.get_mean(disc_fake))
        else:
          disc_fake = input_fake[i][-1]
          disc_real = input_real[i][-1]
          cost += (self.get_mean(disc_fake) - self.get_mean(disc_real))
      return cost
    # for patch_discriminator
    else:
      if is_G:
        disc_fake = input_fake[-1]
        cost = (-self.get_mean(disc_fake))
      else:
        disc_fake = input_fake[-1]
        disc_real = input_real[-1]
        cost = (self.get_mean(disc_fake) - self.get_mean(disc_real))
      return cost

# Restruction Loss
class RestructionLoss(nn.Module):
  '''
  reduction: 'elementwise_mean' or 'none'
  '''
  def __init__(self, distance='l1', reduction='elementwise_mean'):
    super(RestructionLoss, self).__init__()
    if distance == 'l1':
      self.loss = nn.L1Loss(reduction=reduction)
    elif distance == 'mse':
      self.loss = nn.MSELoss(reduction=reduction)
    else:
      raise NotImplementedError()

  def forward(self, gt, pred):
    return self.loss(gt, pred)