# ------------------------------------------------------------------------------
# Copyright (c) Tencent
# Licensed under the GPLv3 License.
# Created by Kai Ma (makai0324@gmail.com)
# ------------------------------------------------------------------------------

from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import os
import os.path

IMG_EXTENSIONS = [
  '.jpg', '.JPG', '.jpeg', '.JPEG',
  '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


def is_image_file(filename):
  return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def make_dataset(dir):
  images = []
  assert os.path.isdir(dir), '%s is not a valid directory' % dir

  for root, _, fnames in sorted(os.walk(dir)):
    for fname in fnames:
      if is_image_file(fname):
        # path = os.path.join(root, fname)
        images.append(fname)

  return images

def get_dataset_from_txt_file(file_path):
  with open(file_path, 'r') as f:
    content = f.readlines()
    return [i.strip() for i in content]

