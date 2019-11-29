# ------------------------------------------------------------------------------
# Copyright (c) Tencent
# Licensed under the GPLv3 License.
# Created by Kai Ma (makai0324@gmail.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


def get_dataset(dataset_name):
  if dataset_name == 'align_ct_xray':
    from .alignDataSet import AlignDataSet
    from .data_augmentation import CT_XRAY_Data_Augmentation, CT_XRAY_Data_Test
    from .collate_fn import collate_gan
    return AlignDataSet, CT_XRAY_Data_Augmentation, CT_XRAY_Data_Test, collate_gan
  elif dataset_name == 'align_ct_xray_views':
    from .alignDataSetViews import AlignDataSet
    from .data_augmentation import CT_XRAY_Data_Augmentation_Multi, CT_XRAY_Data_Test_Multi
    from .collate_fn import collate_gan_views
    return AlignDataSet, CT_XRAY_Data_Augmentation_Multi, CT_XRAY_Data_Test_Multi, collate_gan_views
  elif dataset_name == 'align_ct_xray_baseline':
    from .alignDataSet import AlignDataSet
    from .data_augmentation_baseline import CT_XRAY_Data_Augmentation, CT_XRAY_Data_Test
    from .collate_fn import collate_gan
    return AlignDataSet, CT_XRAY_Data_Augmentation, CT_XRAY_Data_Test, collate_gan
  elif dataset_name == 'align_ct_xray_views_std':
    from .alignDataSetViews import AlignDataSet
    from .data_augmentation_1030 import CT_XRAY_Data_Augmentation_Multi, CT_XRAY_Data_Test_Multi
    from .collate_fn import collate_gan_views
    return AlignDataSet, CT_XRAY_Data_Augmentation_Multi, CT_XRAY_Data_Test_Multi, collate_gan_views
  elif dataset_name == 'align_ct_xray_std':
    from .alignDataSet import AlignDataSet
    from .data_augmentation_1030 import CT_XRAY_Data_Augmentation, CT_XRAY_Data_Test
    from .collate_fn import collate_gan
    return AlignDataSet, CT_XRAY_Data_Augmentation, CT_XRAY_Data_Test, collate_gan
  else:
    raise KeyError('Dataset class should select from align / ')