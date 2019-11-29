# ------------------------------------------------------------------------------
# Copyright (c) Tencent
# Licensed under the GPLv3 License.
# Created by Kai Ma (makai0324@gmail.com)
# ------------------------------------------------------------------------------

from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

from torch.utils.data import Dataset


class Base_DataSet(Dataset):
  '''
  Base DataSet
  '''
  def __int__(self):
    pass

  def __getitem__(self, item):
    return self.pull_item(item)

  def __len__(self):
    return self.num_samples

  @property
  def name(self):
    return 'Base DataSet'

  @property
  def num_samples(self):
    return 1

  def pull_item(self, *str):
    pass

