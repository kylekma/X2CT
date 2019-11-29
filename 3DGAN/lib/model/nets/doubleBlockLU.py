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

'''
Double block linear unit
y = kx when low < x < high
y = low when x <= low
y = high when x >= high
'''


class DoubleBlockLinearUnit(nn.Module):
  def __init__(self, low=0., high=1., k=1.):
    super(DoubleBlockLinearUnit, self).__init__()
    self.low = low
    self.high = high
    self.k = k

  def forward(self, input):
    out = input * 1.0
    out[input > self.high] = self.high
    out[input < self.low] = self.low
    return out

class DoubleBlockUnLinearUnit(nn.Module):
  def __init__(self, low=0., high=1., k=1.):
    super(DoubleBlockUnLinearUnit, self).__init__()
    self.low = 0.
    self.high = 1.
    self.k = k

  def forward(self, input):
    out = torch.pow(input, 2)
    out[input > self.high] = self.high
    out[input < self.low] = self.low
    return out

class LinearUnit(nn.Module):
  def __init__(self):
    super(LinearUnit, self).__init__()

  def forward(self, input):
    return input

class SoftLinearUnit(nn.Module):
  def __init__(self):
    super(SoftLinearUnit, self).__init__()

  def forward(self, input):
    return torch.min(torch.max(0.1*input, input), 0.1*input+0.9)


##################
# Test
##################
def main():
  w = torch.randn(1,4,4)
  print(w)
  activation = DoubleBlockLinearUnit()
  print(activation(w))

if __name__ == '__main__':
  main()
