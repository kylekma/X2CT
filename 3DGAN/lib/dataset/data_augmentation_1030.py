# ------------------------------------------------------------------------------
# Copyright (c) Tencent
# Licensed under the GPLv3 License.
# Created by Kai Ma (makai0324@gmail.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from lib.utils.transform_3d import *
import torch
import numpy as np


def tensor_backto_unnormalization_image(input_image, mean, std):
  '''
  1. image = (image + 1) / 2.0
  2. image = image
  :param input_image: tensor whose size is (c,h,w) and channels is RGB
  :param imtype: tensor type
  :return:
     numpy (c,h,w)
  '''
  if isinstance(input_image, torch.Tensor):
    image_tensor = input_image.data
  else:
    return input_image
  image = image_tensor.data.cpu().float().numpy()
  image = image * std + mean
  return image


class CT_XRAY_Data_Augmentation(object):
  def __init__(self, opt=None):
    self.augment = List_Compose([
      (None, None),

      (Resize_image(size=(opt.ct_channel, opt.fine_size, opt.fine_size)),
       Resize_image(size=(opt.xray_channel, opt.fine_size, opt.fine_size))),

      (Limit_Min_Max_Threshold(opt.CT_MIN_MAX[0], opt.CT_MIN_MAX[1]), None),

      (Normalization(opt.CT_MIN_MAX[0], opt.CT_MIN_MAX[1]),
       Normalization(opt.XRAY1_MIN_MAX[0], opt.XRAY1_MIN_MAX[1])),

      (Normalization_gaussian(opt.CT_MEAN_STD[0], opt.CT_MEAN_STD[1]),
       Normalization_gaussian(opt.XRAY1_MEAN_STD[0], opt.XRAY1_MEAN_STD[1])),

      # (Get_Key_slice(opt.select_slice_num), None),

      (ToTensor(), ToTensor())

    ])

  def __call__(self, img_list):
    '''
    :param img: PIL image
    :param boxes: numpy.ndarray
    :param labels: numpy.ndarray
    :return:
    '''
    return self.augment(img_list)

class CT_XRAY_Data_Test(object):
  def __init__(self, opt=None):
    self.augment = List_Compose([
      (None, None),

      (Resize_image(size=(opt.ct_channel, opt.fine_size, opt.fine_size)),
       Resize_image(size=(opt.xray_channel, opt.fine_size, opt.fine_size))),

      (Limit_Min_Max_Threshold(opt.CT_MIN_MAX[0], opt.CT_MIN_MAX[1]), None),

      (Normalization(opt.CT_MIN_MAX[0], opt.CT_MIN_MAX[1]),
       Normalization(opt.XRAY1_MIN_MAX[0], opt.XRAY1_MIN_MAX[1])),

      (Normalization_gaussian(opt.CT_MEAN_STD[0], opt.CT_MEAN_STD[1]),
       Normalization_gaussian(opt.XRAY1_MEAN_STD[0], opt.XRAY1_MEAN_STD[1])),

      # (Get_Key_slice(opt.select_slice_num), None),

      (ToTensor(), ToTensor())

    ])

  def __call__(self, img):
    '''
    :param img: PIL image
    :param boxes: numpy.ndarray
    :param labels: numpy.ndarray
    :return:
    '''
    return self.augment(img)

class CT_XRAY_Data_AugmentationM(object):
  def __init__(self, opt=None):
    self.augment = List_Compose([
      (Permute((1,0,2)), None),

      (Resize_image(size=(opt.ct_channel, opt.resize_size, opt.resize_size)),
       Resize_image(size=(opt.xray_channel, opt.resize_size, opt.resize_size))),

      (List_Random_cropYX(size=(opt.fine_size, opt.fine_size)),),

      (List_Random_mirror(2), ),

      (Normalization(opt.CT_MIN_MAX[0], opt.CT_MIN_MAX[1]),
       Normalization(opt.XRAY1_MIN_MAX[0], opt.XRAY1_MIN_MAX[1])),

      (Normalization_gaussian(opt.CT_MEAN_STD[0], opt.CT_MEAN_STD[1]),
       Normalization_gaussian(opt.XRAY1_MEAN_STD[0], opt.XRAY1_MEAN_STD[1])),

      # (Get_Key_slice(opt.select_slice_num), None),

      (ToTensor(), ToTensor())

    ])

  def __call__(self, img_list):
    '''
    :param img: PIL image
    :param boxes: numpy.ndarray
    :param labels: numpy.ndarray
    :return:
    '''
    return self.augment(img_list)

class CT_XRAY_Data_TestM(object):
  def __init__(self, opt=None):
    self.augment = List_Compose([
      (Permute((1,0,2)), None),

      (Resize_image(size=(opt.ct_channel, opt.fine_size, opt.fine_size)),
       Resize_image(size=(opt.xray_channel, opt.fine_size, opt.fine_size))),

      (Normalization(opt.CT_MIN_MAX[0], opt.CT_MIN_MAX[1]),
       Normalization(opt.XRAY1_MIN_MAX[0], opt.XRAY1_MIN_MAX[1])),

      (Normalization_gaussian(opt.CT_MEAN_STD[0], opt.CT_MEAN_STD[1]),
       Normalization_gaussian(opt.XRAY1_MEAN_STD[0], opt.XRAY1_MEAN_STD[1])),

      # (Get_Key_slice(opt.select_slice_num), None),

      (ToTensor(), ToTensor())

    ])

  def __call__(self, img):
    '''
    :param img: PIL image
    :param boxes: numpy.ndarray
    :param labels: numpy.ndarray
    :return:
    '''
    return self.augment(img)

class CT_XRAY_Data_Augmentation_Multi(object):
  def __init__(self, opt=None):
    self.augment = List_Compose([
      (None, None, None),

      (Resize_image(size=(opt.ct_channel, opt.fine_size, opt.fine_size)),
       Resize_image(size=(opt.xray_channel, opt.fine_size, opt.fine_size)),
       Resize_image(size=(opt.xray_channel, opt.fine_size, opt.fine_size)),),

      (Limit_Min_Max_Threshold(opt.CT_MIN_MAX[0], opt.CT_MIN_MAX[1]), None, None),

      (Normalization(opt.CT_MIN_MAX[0], opt.CT_MIN_MAX[1]),
       Normalization(opt.XRAY1_MIN_MAX[0], opt.XRAY1_MIN_MAX[1]),
       Normalization(opt.XRAY2_MIN_MAX[0], opt.XRAY2_MIN_MAX[1])),

      (Normalization_gaussian(opt.CT_MEAN_STD[0], opt.CT_MEAN_STD[1]),
       Normalization_gaussian(opt.XRAY1_MEAN_STD[0], opt.XRAY1_MEAN_STD[1]),
       Normalization_gaussian(opt.XRAY2_MEAN_STD[0], opt.XRAY2_MEAN_STD[1])),

      # (Get_Key_slice(opt.select_slice_num), None, None),

      (ToTensor(), ToTensor(), ToTensor())

    ])

  def __call__(self, img_list):
    '''
    :param img: PIL image
    :param boxes: numpy.ndarray
    :param labels: numpy.ndarray
    :return:
    '''
    return self.augment(img_list)

class CT_XRAY_Data_Test_Multi(object):
  def __init__(self, opt=None):
    self.augment = List_Compose([
      (None, None, None),

      (Resize_image(size=(opt.ct_channel, opt.fine_size, opt.fine_size)),
       Resize_image(size=(opt.xray_channel, opt.fine_size, opt.fine_size)),
       Resize_image(size=(opt.xray_channel, opt.fine_size, opt.fine_size)),),

      (Limit_Min_Max_Threshold(opt.CT_MIN_MAX[0], opt.CT_MIN_MAX[1]), None, None),

      (Normalization(opt.CT_MIN_MAX[0], opt.CT_MIN_MAX[1]),
       Normalization(opt.XRAY1_MIN_MAX[0], opt.XRAY1_MIN_MAX[1]),
       Normalization(opt.XRAY2_MIN_MAX[0], opt.XRAY2_MIN_MAX[1])),

      (Normalization_gaussian(opt.CT_MEAN_STD[0], opt.CT_MEAN_STD[1]),
       Normalization_gaussian(opt.XRAY1_MEAN_STD[0], opt.XRAY1_MEAN_STD[1]),
       Normalization_gaussian(opt.XRAY2_MEAN_STD[0], opt.XRAY2_MEAN_STD[1])),

      # (Get_Key_slice(opt.select_slice_num), None),

      (ToTensor(), ToTensor(), ToTensor())

    ])

  def __call__(self, img):
    '''
    :param img: PIL image
    :param boxes: numpy.ndarray
    :param labels: numpy.ndarray
    :return:
    '''
    return self.augment(img)


'''
Data Augmentation
'''
class CT_Data_Augmentation(object):
  def __init__(self, opt=None):
    self.augment = Compose([
      Permute((1,0,2)),
      Normalization(opt.CT_MIN_MAX[0], opt.CT_MIN_MAX[1]),
      Normalization_gaussian(opt.CT_MEAN_STD[0], opt.CT_MEAN_STD[1]),
      Get_Key_slice(opt.select_slice_num),
      ToTensor()
    ])

  def __call__(self, img):
    '''
    :param img: PIL image
    :param boxes: numpy.ndarray
    :param labels: numpy.ndarray
    :return:
    '''
    return self.augment(img)

class Xray_Data_Augmentation(object):
  def __init__(self, opt=None):
    self.augment = Compose([
      Normalization(opt.XRAY1_MIN_MAX[0], opt.XRAY1_MIN_MAX[1]),
      Normalization_gaussian(opt.XRAY1_MEAN_STD[0], opt.XRAY1_MEAN_STD[1]),
      ToTensor()
    ])

  def __call__(self, img):
    '''
    :param img: PIL Image
    :return:
    '''
    return self.augment(img)

class CT_Data_Test(object):
  def __init__(self, opt=None):
    self.augment = Compose([
      Permute((1, 0, 2)),
      Normalization(opt.CT_MIN_MAX[0], opt.CT_MIN_MAX[1]),
      Normalization_gaussian(opt.CT_MEAN_STD[0], opt.CT_MEAN_STD[1]),
      Get_Key_slice(opt.select_slice_num),
      ToTensor()
    ])

  def __call__(self, img):
    '''
    :param img: PIL image
    :param boxes: numpy.ndarray
    :param labels: numpy.ndarray
    :return:
    '''
    return self.augment(img)

class Xray_Data_Test(object):
  def __init__(self, opt=None):
    self.augment = Compose([
      Normalization(opt.XRAY1_MIN_MAX[0], opt.XRAY1_MIN_MAX[1]),
      Normalization_gaussian(opt.XRAY1_MEAN_STD[0], opt.XRAY1_MEAN_STD[1]),
      ToTensor()
    ])

  def __call__(self, img):
    '''
    :param img: PIL image
    :param boxes: numpy.ndarray
    :param labels: numpy.ndarray
    :return:
    '''
    return self.augment(img)

# ##########################################
# Test
# ##########################################
def main():
  test_file = r'D:\Data\LIDC-HDF5-256\LIDC-IDRI-0001.20000101.3000566.1\ct_xray_data.h5'
  import h5py
  import matplotlib.pyplot as plt

  from lib.config.config import cfg, merge_dict_and_yaml
  opt = merge_dict_and_yaml(dict(), cfg)

  hdf = h5py.File(test_file, 'r')
  ct = np.asarray(hdf['ct'])
  xray = np.asarray(hdf['xray1'])
  xray = np.expand_dims(xray, 0)
  print(xray.shape)
  transforma = CT_XRAY_Data_Augmentation(opt)
  transform_normal = CT_XRAY_Data_Test(opt)
  ct_normal, xray_normal = transform_normal([ct, xray])
  ct_trans, xray_trans = transforma([ct, xray])
  ct_trans = tensor_backto_unnormalization_image(ct_trans, opt.CT_MEAN_STD[0], opt.CT_MEAN_STD[1])
  xray_trans = tensor_backto_unnormalization_image(xray_trans, opt.XRAY1_MEAN_STD[0], opt.XRAY1_MEAN_STD[1])
  ct_normal = tensor_backto_unnormalization_image(ct_normal, opt.CT_MEAN_STD[0], opt.CT_MEAN_STD[1])
  xray_normal = tensor_backto_unnormalization_image(xray_normal, opt.XRAY1_MEAN_STD[0], opt.XRAY1_MEAN_STD[1])
  bb = Normalization_to_range()
  ct_trans = bb(ct_trans)
  xray_trans = bb(xray_trans)
  # trans_CT = CT_Data_Augmentation(opt)
  # trans_Xray = Xray_Data_Augmentation(opt)
  # ct_trans = trans_CT(ct).numpy()
  # xray_trans = trans_Xray(xray).numpy()
  import cv2
  print(ct_trans.shape, ct_normal.shape)
  cv2.imshow('1', xray_trans[0].astype(np.uint8))
  cv2.imshow('2', ct_trans[80, :, :].astype(np.uint8))
  cv2.imshow('1-1', bb(xray_normal)[0].astype(np.uint8))
  cv2.imshow('2-1', bb(ct_normal)[80, :, :].astype(np.uint8))
  cv2.waitKey(0)
  # plt.figure(1)
  # plt.imshow(xray_trans[0], cmap=plt.cm.bone)
  # plt.figure(2)
  # plt.imshow(ct_trans[80, :, :], cmap=plt.cm.bone)
  # plt.show()

if __name__ == '__main__':
  main()
