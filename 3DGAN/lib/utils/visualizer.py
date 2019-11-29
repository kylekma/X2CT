# ------------------------------------------------------------------------------
# Copyright (c) Tencent
# Licensed under the GPLv3 License.
# Created by Kai Ma (makai0324@gmail.com)
# ------------------------------------------------------------------------------

from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import tensorboardX as tbX
import numpy as np
import os
import torch
import torchvision.utils as vutils
import cv2


def tensor_to_image(tensor, imtype=np.uint8):
  '''
  :param tensor:
    (c,h,w)
  :return:
  '''
  img = tensor.cpu().numpy()
  img = np.transpose(img, (1,2,0))

  if np.uint8 == imtype:
    if np.max(img) > 1:
      print(np.max(img))
      raise ValueError('Image value should range from 0 to 1.')
    img = img * 255.0

  return img.astype(imtype)


def save_image(image_numpy, image_path):
  image_pil = image_numpy
  cv2.imwrite(image_path, image_pil)

# save image to the disk
def save_images(webpage, visuals, image_path, normalize_list, aspect_ratio=1.0, width=256, max_image=10):
  image_dir = webpage.get_image_dir()
  # short_path1 = os.path.basename(image_path[0][0])
  # short_path2 = os.path.basename(image_path[1][0])
  # name1 = os.path.splitext(short_path1)[0]
  # name2 = os.path.splitext(short_path2)[0]
  # name = name1+'To'+name2
  # short_path = os.path.split(image_path[0][0])
  name1 = os.path.splitext(os.path.basename(image_path[0][0]))[0]
  name2 = os.path.split(os.path.dirname(image_path[0][0]))[-1]
  name = name2+'_'+name1

  webpage.add_header(name)

  image_root = os.path.join(image_dir, name)
  if not os.path.exists(image_root):
    os.makedirs(image_root)

  count = 0
  for label, image_tensor in visuals.items():
    image_tensor = image_tensor.data.clone().cpu()[0]
    image_list = add_3D_image(image_tensor, max_image)
    image_list = [tensor_back_to_unnormalization(img, normalize_list[count][0], normalize_list[count][1]) for img in image_list]
    ims, txts, links = [], [], []
    count += 1

    for image_i, image_t in enumerate(image_list):
      im = tensor_to_image(image_t)
      image_name = '%s_%s_%d.png' % (name, label, image_i)

      save_path = os.path.join(image_root, image_name)
      h, w, _ = im.shape
      # if aspect_ratio > 1.0:
      #     im = imresize(im, (h, int(w * aspect_ratio)), interp='bicubic')
      # if aspect_ratio < 1.0:
      #     im = imresize(im, (int(h / aspect_ratio), w), interp='bicubic')
      save_image(im, save_path)

      ims.append(os.path.join(name, image_name))
      txts.append(label)
      links.append(os.path.join(name, image_name))
    webpage.add_images(ims, txts, links, width=width)

def add_3D_image(tensor, max_image):
  '''
  :param tensor:
    (c,h,w)
  :param max_image:
  :return:
  '''
  c, h, w = tensor.size()
  if c <= max_image:
    images = [tensor[i:i+1] for i in range(c)]
  else:
    skip_len = float(c) / max_image
    images = [tensor[int(i*skip_len):(int(i*skip_len) + 1)] for i in range(max_image)]

  return images

def tensor_back_to_unnormalization(input_image, mean, std):
  image = input_image * std + mean
  return image

def tensor_back_to_unMinMax(input_image, min, max):
  image = input_image * (max - min) + min
  return image

class Visualizer(object):
  '''
  Visual train process using tensorboardX
  '''
  def __init__(self, log_dir):
    self.tb = tbX.SummaryWriter(log_dir=log_dir)
    self.cache = dict()

  def add_graph(self, model, input):
    self.tb.add_graph(model, input)

  def add_image(self, name, image_dict, normalize_list, step, max_image=10):
    '''
    :param image_dict: {key:tensor}
    :return:
    '''
    count = 0
    for key, image in image_dict.items():
      image_list = add_3D_image(image.data.clone().cpu()[0], max_image)
      image_list = [tensor_back_to_unnormalization(img, normalize_list[count][0], normalize_list[count][1]) for img in image_list]
      image = vutils.make_grid(image_list, normalize=False, scale_each=False)
      self.tb.add_image(name+'_'+str(key), image, global_step=step)
      count += 1

  # single loss write
  def add_scalar(self, name, value, step):
    self.tb.add_scalar(name, value, step)

  # various loss write
  def add_scalars(self, main_tag, tag_dict, step):
    self.tb.add_scalars(main_tag=main_tag, tag_scalar_dict=tag_dict, global_step=step)

  # total loss write
  def add_total_scalar(self, name, tag_dict, step):
    total_loss = float(np.sum([x for x in tag_dict.values()]))
    self.tb.add_scalar(name, total_loss, step)
    return total_loss

  # average loss write
  # 1. add in the inner circle(write=False)
  # 2. calculate in the outside(write=True)
  def add_average_scalar(self, name, value=None, step=None, write=False):
    if value is not None:
      if name in self.cache:
        self.cache[name].append(value)
      else:
        self.cache[name] = [value]
    if write:
      self.tb.add_scalar(name, float(np.average(self.cache[name])), global_step=step)
      # remove tag
      self.cache.pop(name)

  # average loss write
  # 1. add in the inner circle(write=False)
  # 2. calculate in the outside(write=True)
  def add_average_scalers(self, main_tag, tag_dict=None, step=None, write=False):
    if tag_dict is not None:
      if main_tag in self.cache:
        for k, v in tag_dict.items():
          self.cache[main_tag][k].append(v)
      else:
        self.cache[main_tag] = {}
        for k, v in tag_dict.items():
          self.cache[main_tag][k] = [v]
    if write:
      # average value calculating
      moving_dict = {}
      for k, v in self.cache[main_tag].items():
        moving_dict[k] = float(np.average(v))
      self.tb.add_scalars(main_tag=main_tag, tag_scalar_dict=moving_dict, global_step=step)
      # remove tag
      self.cache.pop(main_tag)

  # add histogram
  def add_histogram(self, name, value, step):
    self.tb.add_histogram(name, value, step)


def main():
  from lib.dataset.data_augmentation import CT_Data_Augmentation, Xray_Data_Augmentation
  test_file = r'D:\Data\LIDC-HDF5-128\LIDC-IDRI-0001.20000101.3000566.1\ct_xray_data.h5'
  import h5py

  from lib.config.config import cfg, merge_dict_and_yaml
  opt = merge_dict_and_yaml(dict(), cfg)

  hdf = h5py.File(test_file, 'r')
  ct = np.asarray(hdf['ct'])
  xray = np.asarray(hdf['xray1'])
  xray = np.expand_dims(xray, 0)
  print(xray.shape)
  trans_CT = CT_Data_Augmentation(opt)
  trans_Xray = Xray_Data_Augmentation(opt)
  ct_trans = trans_CT(ct)
  xray_trans = trans_Xray(xray)

  visual_dict = {'ct': torch.unsqueeze(ct_trans, 0),
                 'xray': torch.unsqueeze(xray_trans, 0)}
  visual = Visualizer(log_dir='../../demo/log')
  visual.add_image('a', visual_dict, 1)
  visual.add_scalar('b', 1, 1)

if __name__ == '__main__':
  main()