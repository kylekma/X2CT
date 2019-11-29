# ------------------------------------------------------------------------------
# Copyright (c) Tencent
# Licensed under the GPLv3 License.
# Created by Kai Ma (makai0324@gmail.com)
# ------------------------------------------------------------------------------

from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import numpy as np
import torch
import scipy.ndimage as ndimage

class Compose(object):

  def __init__(self, transforms):
    self.transforms = transforms

  def __call__(self, img):
    for t in self.transforms:
      img = t(img)
    return img


class List_Compose(object):
  def __init__(self, transforms):
    self.transforms = transforms

  def __call__(self, img_list):
    for t_list in self.transforms:
      # deal with separately
      if len(t_list) > 1:
        new_img_list = []
        for img, t in zip(img_list, t_list):
          if t is None:
            new_img_list.append(img)
          else:
            new_img_list.append(t(img))
        img_list = new_img_list
      # deal with combined
      else:
        img_list = t_list[0](img_list)

    return img_list


def _isArrayLike(obj):
  return hasattr(obj, '__iter__') and hasattr(obj, '__len__')


class List_Random_mirror(object):
  '''
    Returns:
      img: 3d array, (z,y,x) or (D, H, W)
  '''
  def __init__(self, index):
    self.index = index

  def __call__(self, img_list):
    if np.random.random() < 0.5:
      transformed_img_list = []
      for img in img_list:
        if self.index == 0:
          img_copy = img[::-1, :, :]
        elif self.index == 1:
          img_copy = img[:, ::-1, :]
        elif self.index == 2:
          img_copy = img[:, :, ::-1]
        else:
          raise ValueError()
        transformed_img_list.append(img_copy)
      return transformed_img_list
    else:
      return img_list

class List_Random_cropYX(object):
  '''
    Returns:
      img: 3d array, (z,y,x) or (D, H, W)
  '''
  def __init__(self, size=(256,256)):
    if not _isArrayLike(size):
      raise ValueError('each dimension of size must be defined')
    self.size = np.array(size, dtype=np.int)

  def __call__(self, img_list):
    for img in img_list:
      for img_b in img_list:
        assert img.shape[1:] == img_b.shape[1:]
    _, y, x = img_list[0].shape
    h, w = self.size
    assert y - h >= 0, 'crop size is bigger than image size'
    assert x - w >= 0, 'crop size is bigger than image size'
    transformed_img_list = []
    if np.random.random(1) < 0.5:
      i = np.random.randint(0, y - h)
      j = np.random.randint(0, x - w)
      for img in img_list:
        img_copy = img[:, i:i+h, j:j+w]
        transformed_img_list.append(img_copy)
    else:
      target_size = self.size.astype(np.float32)
      for img in img_list:
        img_copy = ndimage.interpolation.zoom(img, (1., target_size[0] / y, target_size[1] / x), order=1)
        transformed_img_list.append(img_copy)

    return transformed_img_list


class Resize_image(object):
  '''
    Returns:
      img: 3d array, (z,y,x) or (D, H, W)
  '''
  def __init__(self, size=(3,256,256)):
    if not _isArrayLike(size):
      raise ValueError('each dimension of size must be defined')
    self.size = np.array(size, dtype=np.float32)

  def __call__(self, img):
    z, x, y = img.shape
    ori_shape = np.array((z, x, y), dtype=np.float32)
    resize_factor = self.size / ori_shape
    img_copy = ndimage.interpolation.zoom(img, resize_factor, order=1)

    return img_copy

class Random_mirror(object):
  '''
    Returns:
      img: 3d array, (z,y,x) or (D, H, W)
  '''
  def __init__(self, index):
    self.index = index

  def __call__(self, img):
    if np.random.random() < 0.5:
      if self.index == 0:
        img = img[::-1, :, :]
      elif self.index == 1:
        img = img[:, ::-1, :]
      elif self.index == 2:
        img = img[:, :, ::-1]

    return img

# class Resize(object):
#   '''
#     Returns:
#       img: 3d array, (z,y,x) or (D, H, W)
#   '''
#   def __init__(self, shape):
#     if not _isArrayLike(shape):
#       shape = np.array((shape, shape, shape), dtype=np.float32)
#     self.shape = shape
#
#   def __call__(self, img):
#     ori_shape = img.shape
#     new_shape = self.shape / ori_shape
#     img = ndimage.zoom(img, new_shape, mode='nearest')
#
#     return img

class Permute(object):
  '''
  Permute
    Returns:
      img: 3d array, (z,y,x) or (D, H, W)
  '''
  def __init__(self, loc):
    self.loc = loc

  def __call__(self, img):
    img = np.transpose(img, self.loc)

    return img

class ToTensor(object):
  '''
  To Torch Tensor
  img: 3D, (z, y, x) or (D, H, W)
    Returns:
      img: 3d array, (z,y,x) or (D, H, W)
  '''
  def __call__(self, img):
    img = torch.from_numpy(img.astype(np.float32))

    return img

class Normalization(object):
  '''
  To value range 0-1
  img: 3D, (z, y, x) or (D, H, W)
    Returns:
      img: 3d array, (z,y,x) or (D, H, W)
  '''
  def __init__(self, min, max, round_v=6):
    '''
    :param min:
    :param max:
    :param round_v:
      decrease calculating time
    '''
    self.range = np.array((min, max), dtype=np.float32)
    self.round_v = round_v

  def __call__(self, img):
    img_copy = img.copy()
    img_copy = np.round((img_copy - self.range[0]) / (self.range[1] - self.range[0]), self.round_v)

    return img_copy


class Limit_Min_Max_Threshold(object):
  '''
  Restrict in value range. value > max = max,
  value < min = min
  img: 3D, (z, y, x) or (D, H, W)
    Returns:
      img: 3d array, (z,y,x) or (D, H, W)
  '''
  def __init__(self, min, max):
    self.min = min
    self.max = max

  def __call__(self, img):
    img_copy = img.copy()
    img_copy[img_copy > self.max] = self.max
    img_copy[img_copy < self.min] = self.min

    return img_copy


class Normalization_gaussian(object):
  '''
  To value range 0-1
  img: 3D, (z, y, x) or (D, H, W)
    Returns:
      img: 3d array, (z,y,x) or (D, H, W)
  '''
  def __init__(self, mean, std):
    self.mean = mean
    self.std = std

  def __call__(self, img):
    img_copy = img.copy()
    img_copy = (img_copy - self.mean) / self.std

    return img_copy

class Normalization_to_range(object):
  '''
    Must range 0-1 first!
    To value specific range
    img: 3D, (z, y, x) or (D, H, W)
      Returns:
        img: 3d array, (z,y,x) or (D, H, W)
  '''
  def __init__(self, min=0, max=255):
    self.range = np.array((min, max), dtype=np.float32)

  def __call__(self, img):
    img_copy = img.copy()
    img_copy = img_copy * (self.range[1] - self.range[0]) + self.range[0]

    return img_copy

class Get_Key_slice(object):
  '''
    get specific slice from volume
    img: 3D, (z, y, x) or (D, H, W)
      Returns:
        img: 3d array, (z,y,x) or (D, H, W)
  '''
  def __init__(self, aim_num):
    self.aim_num = aim_num

  def __call__(self, img):
    # without any dispose
    if self.aim_num == 0:
      return img

    d, _, _ = img.shape

    if d <= self.aim_num:
      raise ValueError('aim_num is larger than the first dimension of image')

    block_len = np.floor(np.divide(d, self.aim_num))
    select_index = []
    for i in range(self.aim_num):
      begin = block_len * i
      end = block_len * (i + 1)
      mid = np.floor(np.divide(begin+end, 2))
      select_index.append(int(mid))
    img = img[select_index]
    return img


