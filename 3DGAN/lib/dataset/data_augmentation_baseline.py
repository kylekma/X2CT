# ------------------------------------------------------------------------------
# Copyright (c) Tencent
# Licensed under the GPLv3 License.
# Created by Kai Ma (makai0324@gmail.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from lib.utils.transform_3d import *


class CT_XRAY_Data_Augmentation(object):
  def __init__(self, opt=None):
    self.augment = List_Compose([
      (None, None),

      (Resize_image(size=(opt.ct_channel, opt.fine_size, opt.fine_size)),
       Resize_image(size=(opt.xray_channel, opt.fine_size*2, opt.fine_size*2))),

      (Limit_Min_Max_Threshold(opt.CT_MIN_MAX[0], opt.CT_MIN_MAX[1]), None),

      (Normalization(opt.CT_MIN_MAX[0], opt.CT_MIN_MAX[1]),
       Normalization(opt.XRAY1_MIN_MAX[0], opt.XRAY1_MIN_MAX[1])),

      (Normalization_gaussian(opt.CT_MEAN_STD[0], opt.CT_MEAN_STD[1]),
       Normalization_gaussian(opt.XRAY1_MEAN_STD[0], opt.XRAY1_MEAN_STD[1])),

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
       Resize_image(size=(opt.xray_channel, opt.fine_size*2, opt.fine_size*2))),

      (Limit_Min_Max_Threshold(opt.CT_MIN_MAX[0], opt.CT_MIN_MAX[1]), None),

      (Normalization(opt.CT_MIN_MAX[0], opt.CT_MIN_MAX[1]),
       Normalization(opt.XRAY1_MIN_MAX[0], opt.XRAY1_MIN_MAX[1])),

      (Normalization_gaussian(opt.CT_MEAN_STD[0], opt.CT_MEAN_STD[1]),
       Normalization_gaussian(opt.XRAY1_MEAN_STD[0], opt.XRAY1_MEAN_STD[1])),

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