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
    self.template = self.init_template()

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


  def init_template(self):
    #Take first traning CT scan from data/train.txt make sure it doesn't reapper in test
    #Talk to Wu or someone else and see if I should remove it for training not as 3rd input but as input to reconstruction loss training error function
    #could be as easy as removing LIDC-IDRI-0936.20000101.5169.2.1 from train.txt and ofc test.txt if it is there for some reason
    #It is not there
    file_path = os.path.join("./data/LIDC-HDF5-256","LIDC-IDRI-0936.20000101.5169.2.1",'ct_xray_data'+self.ext)
    hdf5 = h5py.File(file_path, 'r')
    ct_data = np.asarray(hdf5['ct'])
    hdf5.close()
    return ct_data


  def get_image_path(self, root, index_name):
    print("root = " + str(root) + "/n")
    print("index= " + str(index_name))
    img_path = os.path.join(root, index_name, 'ct_xray_data'+self.ext)
    assert os.path.exists(img_path), 'Path do not exist: {}'.format(img_path)
    return img_path





  def load_file(self, file_path):
    #assume that 0th index for ct and xray files refer to same person, doublecheck
    #problem is that there are only 1 och each xray here and 256 isch CT scans when we load
    #problem 2, given that there are only 2 xrays make sure that we do not reconstruct xrays thorugh DDR on the template CT
    #Solution: remove x-rays from template. Only take out 1 CT scan and remove CT scan from training data.

    print(file_path)

    hdf5 = h5py.File(file_path, 'r')
    ct_data = np.asarray(hdf5['ct'])

    
    x_ray1 = np.asarray(hdf5['xray1'])
    x_ray2 = np.asarray(hdf5['xray2'])

    x_ray1 = np.expand_dims(x_ray1, 0)
    x_ray2 = np.expand_dims(x_ray2, 0)
    #extract ct one for person that will act as template



    hdf5.close()
    return ct_data, x_ray1, x_ray2

    


  '''
  generate batch
  '''
  def pull_item(self, item):
    file_path = self.get_image_path(self.dir_root, self.dataset_paths[item]) 
    ct_data, x_ray1, x_ray2 = self.load_file(file_path)
    #print(template_data)
    #for idx, e in enumerate(x_ray2):
    #    print("Xray " + str(idx) + "\n")
    #    print(np.shape(e))
    #    break
    #print("LENGTH X1: "+ str(len(x_ray1)) +" LENGTH X2 "+ str(len(x_ray2)) + " LENGTH CT: " + str(len(ct_data)))

    # Data Augmentation

    
    ct, xray1, xray2 = self.data_augmentation([ct_data, x_ray1, x_ray2])
    template = self.template


    #Data Augmentation for template
    #template_data["template_ct"],template_data["template_x_ray1"],template_data["template_x_ray2"] = self.data_augmentation([template_data["template_ct"], template_data["template_x_ray1"],template_data["template_x_ray2"]])
  

    return ct, xray1, xray2, file_path, template






