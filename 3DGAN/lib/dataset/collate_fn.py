# ------------------------------------------------------------------------------
# Copyright (c) Tencent
# Licensed under the GPLv3 License.
# Created by Kai Ma (makai0324@gmail.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import torch

def collate_gan(batch):
  '''
  :param batch: [imgs, boxes, labels] dtype = np.ndarray
  imgs:
    shape = (C H W)
  :return:
  '''
  ct = [x[0] for x in batch]
  xray = [x[1] for x in batch]
  file_path = [x[2] for x in batch]

  return torch.stack(ct), torch.stack(xray), file_path

def collate_gan_views(batch):
  '''
  :param batch: [imgs, boxes, labels] dtype = np.ndarray
  imgs:
    shape = (C H W)
  :return:
  '''
  ct = [x[0] for x in batch]
  xray1 = [x[1] for x in batch]
  xray2 = [x[2] for x in batch]
  file_path = [x[3] for x in batch]

  return torch.stack(ct), [torch.stack(xray1), torch.stack(xray2)], file_path