# ------------------------------------------------------------------------------
# Copyright (c) Tencent
# Licensed under the GPLv3 License.
# Created by Kai Ma (makai0324@gmail.com)
# ------------------------------------------------------------------------------

from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

from lib.dataset.baseDataSet import Base_DataSet
from lib.dataset.utils import *
import h5py
import numpy as np

class AlignDataSet(Base_DataSet):
  '''
  DataSet For unaligned data
  '''
  def __init__(self, opt):
    super(AlignDataSet, self).__init__()
    self.opt = opt
    self.ext = '.h5'
    self.dataset_paths = get_dataset_from_txt_file(self.opt.datasetfile)
    self.dataset_paths = sorted(self.dataset_paths)
    self.dataset_size = len(self.dataset_paths)
    self.dir_root = self.get_data_path
    self.data_augmentation = self.opt.data_augmentation(opt)

  @property
  def name(self):
    return 'AlignDataSet'

  @property
  def get_data_path(self):
    path = os.path.join(self.opt.dataroot)
    return path

  @property
  def num_samples(self):
    return self.dataset_size

  def get_image_path(self, root, index_name):
    img_path = os.path.join(root, index_name, 'ct_xray_data'+self.ext)
    assert os.path.exists(img_path), 'Path do not exist: {}'.format(img_path)
    return img_path

  def load_file(self, file_path):
    #assume that 0th index for ct and xray files refer to same person, doublecheck

    hdf5 = h5py.File(file_path, 'r')
    ct_data = np.asarray(hdf5['ct'])

    
    x_ray1 = np.asarray(hdf5['xray1'])
    x_ray2 = np.asarray(hdf5['xray2'])
    x_ray1 = np.expand_dims(x_ray1, 0)
    x_ray2 = np.expand_dims(x_ray2, 0)


    template_ct = ct_data[0] #extract ct one for person that will act as template
    template_x_ray1 = x_ray1[0]
    template_x_ray2 = x_ray2[0]

    x_ray1 = x_ray1[1:len(x_ray1)+1]
    x_ray2 = x_ray2[1:len(x_ray2)+1]
    ct_data = ct_data[1:len(ct_data)+1]
    
    template_data = {"template_ct":template_ct,"template_x_ray1":template_x_ray1,"template_x_ray2":template_x_ray2}
    hdf5.close()
    return ct_data, x_ray1, x_ray2, template_data


  '''
  generate batch
  '''
  def pull_item(self, item):
    file_path = self.get_image_path(self.dir_root, self.dataset_paths[item])
    ct_data, x_ray1, x_ray2 = self.load_file(file_path)

    # Data Augmentation
    ct, xray1, xray2 = self.data_augmentation([ct_data, x_ray1, x_ray2])

    return ct, xray1, xray2, file_path






