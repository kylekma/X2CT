# ------------------------------------------------------------------------------
# Copyright (c) Tencent
# Licensed under the GPLv3 License.
# Created by Kai Ma (makai0324@gmail.com)
# ------------------------------------------------------------------------------

from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import torch
import torch.nn.functional as F


def Mean_Squared_Error(tensorA, tensorB):
  '''
  :param tensorA:
    NCHW
  :param tensorB:
    NCHW
  :return:
  '''
  assert isinstance(tensorA, torch.Tensor) and isinstance(tensorB, torch.Tensor)
  assert tensorA.dim() == 4 and tensorB.dim() == 4

  mse = F.mse_loss(tensorA, tensorB)
  return mse

# def Peak_Signal_to_Noise_Rate(tensorA, tensorB, PIXEL_MAX):
#   '''
#   10*log10(MAX**2/MSE)
#     :param tensorA:
#       NCHW
#     :param tensorB:
#       NCHW
#     :return:
#   '''
#   assert isinstance(tensorA, torch.Tensor) and isinstance(tensorB, torch.Tensor)
#   assert tensorA.dim() == 4 and tensorB.dim() == 4
#   mse = torch.pow(tensorA - tensorB, 2).mean(dim=1).mean(dim=1).mean(dim=1)
#   psnr = 20 * torch.log10(PIXEL_MAX / torch.sqrt(mse))
#   return torch.mean(psnr)

def Peak_Signal_to_Noise_Rate(tensorA, tensorB, PIXEL_MAX):
  '''
  10*log10(MAX**2/MSE)
    :param tensorA:
      NCHW
    :param tensorB:
      NCHW
    :return:
  '''
  assert isinstance(tensorA, torch.Tensor) and isinstance(tensorB, torch.Tensor)
  assert tensorA.dim() == 4 and tensorB.dim() == 4
  tensorA_copy = tensorA.view(tensorA.size(0), -1)
  tensorB_copy = tensorB.view(tensorB.size(0), -1)
  mse = torch.pow(tensorA_copy - tensorB_copy, 2).mean(dim=1)
  psnr = 20 * torch.log10(PIXEL_MAX / torch.sqrt(mse))
  return torch.mean(psnr)

def Cosine_Similarity(tensorA, tensorB):
  '''
  :param tensorA:
    NCHW
  :param tensorB:
    NCHW
  :return:
  '''
  assert isinstance(tensorA, torch.Tensor) and isinstance(tensorB, torch.Tensor)
  assert tensorA.dim() == 4 and tensorB.dim() == 4
  tensorA_copy = tensorA.view(tensorA.size(0), -1)
  tensorB_copy = tensorB.view(tensorB.size(0), -1)

  cosineV = torch.mean(F.cosine_similarity(tensorA_copy, tensorB_copy))
  return cosineV

# def Wassertein_Distance(tensorA, tensorB):
#   '''
#     :param tensorA:
#       NCHW gt
#     :param tensorB:
#       NCHW prediction
#     :return:
#     '''
#   assert isinstance(tensorA, torch.Tensor) and isinstance(tensorB, torch.Tensor)
#   assert tensorA.dim() == 4 and tensorB.dim() == 4
#
#   cosineV = torch.mean(tensorA) - torch.mean(tensorB)
#
#   return cosineV

# def Structural_Similarity(tensorA, tensorB):
#   '''
#   :param tensorA:
#     NCHW
#   :param tensorB:
#     NCHW
#   :return:
#   '''
#   assert isinstance(tensorA, torch.Tensor) and isinstance(tensorB, torch.Tensor)
#   assert tensorA.dim() == 4 and tensorB.dim() == 4
#
#   return ssim(tensorA,tensorB)

'''
Python
'''


###############################################
# Test
###############################################

def main():
  import numpy as np
  import sklearn.metrics as Metrics
  from sklearn.metrics.pairwise import cosine_similarity
  a = np.random.randint(0, 256,size=(2,3,10,10))
  b = np.random.randint(0, 256, size=(2,3,10,10))
  tensor_a = torch.from_numpy(a.astype(np.float64))
  tensor_b = torch.from_numpy(b.astype(np.float64))
  # print(Peak_Signal_to_Noise_Rate(tensor_a, tensor_b, 255), Peak_Signal_to_Noise_Rate1(tensor_a, tensor_b, 255))
  # print(Mean_Squared_Error(tensor_a, tensor_b))
  # w = np.mean(np.power(a-b, 2))
  # print(w)
  #
  # print(Cosine_Similarity(tensor_a, tensor_b))
  #
  # for i in range(2):
  #   print(cosine_similarity(a[i].reshape(1,-1), b[i].reshape(1,-1)))


if __name__ == '__main__':
  main()