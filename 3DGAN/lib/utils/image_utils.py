# ------------------------------------------------------------------------------
# Copyright (c) Tencent
# Licensed under the GPLv3 License.
# Created by Kai Ma (makai0324@gmail.com)
# ------------------------------------------------------------------------------

import cv2


def im_read(path, color='rgb'):
  '''
  :param path:
  :param color:
    'rgb', 'bgr', 'gray'
  :return:
  '''
  img = cv2.imread(path)
  if color == 'rgb':
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
  elif color == 'bgr':
    pass
  elif color == 'gray':
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  else:
    raise NotImplementedError('Must choose from rgb, bgr, gray')
  return img