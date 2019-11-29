# ------------------------------------------------------------------------------
# Copyright (c) Tencent
# Licensed under the GPLv3 License.
# Created by Kai Ma (makai0324@gmail.com)
# ------------------------------------------------------------------------------

from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import torch.nn as nn


'''
Map loss
'''
# Because of normalization process before network input, it's
# necessary to do some transition to make consistent between
# prediction and ground truth
class Map_loss(nn.Module):
  def __init__(self, direct_mean='l1', predict_transition=None, gt_transition=None):
    super(Map_loss, self).__init__()
    self.direct_mean = direct_mean
    self.predict_transition = predict_transition
    self.gt_transition = gt_transition
    if direct_mean == 'l1':
      self.loss = nn.L1Loss()
    elif direct_mean == 'mse':
      self.loss = nn.MSELoss()
    elif direct_mean == 'kl':
      self.loss = nn.KLDivLoss()
    else:
      raise ValueError()

  # # Min-Max transition
  # def transition(self, predict):
  #   new_predict = (predict * (self.predict_transition[1]-self.predict_transition[0])
  #                  + self.predict_transition[0] -self.gt_transition[0])\
  #                 / (self.gt_transition[1]-self.gt_transition[0])
  #   return new_predict

  def forward(self, predict, gt):
    '''
    :param input:
      [predict, gt]
    :return:
    '''
    if self.direct_mean == 'kl':
      return self.loss(predict, gt)
    else:
      # new_predict = self.transition(predict)
      return self.loss(predict, gt)