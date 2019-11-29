# ------------------------------------------------------------------------------
# Copyright (c) Tencent
# Licensed under the GPLv3 License.
# Created by Kai Ma (makai0324@gmail.com)
# ------------------------------------------------------------------------------

import SimpleITK as sitk


class CTVisual(object):
  def __init__(self):
    pass

  def save(self, volume, spacing, origin, path):
    itkimage = sitk.GetImageFromArray(volume, isVector=False)
    itkimage.SetSpacing(spacing)
    itkimage.SetOrigin(origin)
    sitk.WriteImage(itkimage, path, True)

  def load_scan_mhd(self, path):
    img_itk = sitk.ReadImage(path)
    img = sitk.GetArrayFromImage(img_itk)
    return img_itk, img, img_itk.GetSpacing(), img_itk.GetOrigin()

  def inverse_z_direction(self, ct_scan):
    return ct_scan[::-1, :, :]

  def ct_transform(self, ct_path):
    ct_itk, ct_scans, ori_spacing, ori_origin = self.load_scan_mhd(ct_path)
    # original CT is inversed in slice dimension
    ct_scans_standard_same_shape = self.inverse_z_direction(ct_scans)
    return ct_scans_standard_same_shape, ori_spacing, ori_origin