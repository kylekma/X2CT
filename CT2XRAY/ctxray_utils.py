# ------------------------------------------------------------------------------
# Copyright (c) Tencent
# Licensed under the GPLv3 License.
# Created by Kai Ma (makai0324@gmail.com)
# ------------------------------------------------------------------------------
import os
import cv2
import time
import pydicom
import numpy as np
import SimpleITK as sitk

# Load the scans in given folder path
def load_scan_dcm(path):
  slices = [pydicom.dcmread(os.path.join(path, s)) for s in os.listdir(path) if 'dcm' in s]
  slices.sort(key=lambda x: int(x.ImagePositionPatient[2]))

  try:
    slice_thickness = np.abs(slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2])
  except:
    slice_thickness = np.abs(slices[0].SliceLocation - slices[1].SliceLocation)

  for s in slices:
    if hasattr(s, 'SliceThickness'):
      pass
    else:
      s.SliceThickness = slice_thickness
  return slices

# input could be .mhd/.mha format
def load_scan_mhda(path):
    img_itk = sitk.ReadImage(path)
    img = sitk.GetArrayFromImage(img_itk)
    return img_itk, img, img_itk.GetOrigin(), img_itk.GetSize(), img_itk.GetSpacing()

# output coule be .mhd/.mha format
def save_scan_mhda(volume, origin, spacing, path):
    itkimage = sitk.GetImageFromArray(volume, isVector=False)
    itkimage.SetSpacing(spacing)
    itkimage.SetOrigin(origin)
    sitk.WriteImage(itkimage, path, True)
    
# do betstriping
def bedstriping(ct_scan, ct_mask):
    bedstrip_scan = ct_scan * ct_mask
    return bedstrip_scan

# build CycleGAN trainSet A
def buildTrainA(raw_root_path, save_root_path):
    files_list = os.listdir(raw_root_path)
    if not os.path.exists(save_root_path):
        os.makedirs(save_root_path)
    start = time.time()
    for fileIndex, fileName in enumerate(files_list):
        t0 = time.time()
        print('Begin {}/{}: {}'.format(fileIndex+1, len(files_list), fileName))
        imageDir = os.path.join(raw_root_path, fileName)
        imagePath = os.path.join(imageDir, 'xray1.png')
        image = cv2.imread(imagePath, 0)
        savePath = os.path.join(save_root_path, '{}.png'.format(fileName))
        cv2.imwrite(savePath, image)
        t1 = time.time()
        print('End! Case time: {}'.format(t1-t0))
    end = time.time()
    print('Finally! Total time of build TrainA: {}'.format(end-start))
    
# build CycleGAN trainSet B
def buildTrainB(raw_root_path, save_root_path):
    BSize = 1000
    files_list = os.listdir(raw_root_path)
    if not os.path.exists(save_root_path):
        os.makedirs(save_root_path)
    tb = np.arange(0, len(files_list))
    np.random.shuffle(tb)
    start = time.time()
    for i in range(0, BSize):
        t0 = time.time()
        imageName = files_list[tb[i]]
        print('Begin {}/{}: {}'.format(i+1, BSize, imageName))
        imagePath = os.path.join(raw_root_path, imageName)
        image = cv2.imread(imagePath, 0)
        image = cv2.resize(image, (320, 320))
        savePath = os.path.join(save_root_path, imageName)
        cv2.imwrite(savePath, image)
        t1 = time.time()
        print('End! Case time: {}'.format(t1-t0))
    end = time.time()
    print('Finlly! Total time of build TrainB: {}'.format(end-start))    

# resize all images in a directory
def resize_to_standard(path):
    images = os.listdir(path)
    for i in images:
        imageName = os.path.join(path, i)
        image = cv2.imread(imageName, 0)
        image = cv2.resize(image, (320, 320))
        cv2.imwrite(imageName, image)
    print('All images have been resized!')
    
def convert2hu(path):
    img, scan, origin, size, spacing = load_scan_mhda(path)
    #print(scan.dtype)
    scan = scan.astype(np.int16)
    scanhu = scan - 1024
    root, file = os.path.split(path)
    fileName = file.replace('.mha', 'hu')
    #print(root, file)
    savePath = os.path.join(root, fileName+'.mha') 
    save_scan_mhda(scanhu, origin, spacing, savePath)
    
def psnr(img_1, img_2, PIXEL_MAX = 255.0):
    mse = np.mean((img_1 - img_2) ** 2) 
    if mse == 0:
        return 100
    return 20 * np.log10(PIXEL_MAX / np.sqrt(mse))


if __name__ == '__main__':
    raw_root_pathA = 'D:/LIDC_TEST'
    save_root_pathA = 'D:/xrayv2r/trainA'
    raw_root_pathB =  'F:/Data/NIHCXR/images_002/images'
    save_root_pathB = 'D:/xrayv2r/trainB'
    #buildTrainA(raw_root_pathA, save_root_pathA)
    #buildTrainB(raw_root_pathB, save_root_pathB)

    '''
    raw_root_pathA = 'D:/xrayv2r/trainA500'
    ASize = 260
    files_list = os.listdir(raw_root_pathA)
    if not os.path.exists(save_root_pathA):
        os.makedirs(save_root_pathA)
    start = time.time()
    for i in range(0, ASize):
        t0 = time.time()
        imageName = files_list[i]
        print('Begin {}/{}: {}'.format(i+1, ASize, imageName))
        imagePath = os.path.join(raw_root_pathA, imageName)
        image = cv2.imread(imagePath, 0)
        #image = cv2.resize(image, (320, 320))
        savePath = os.path.join(save_root_pathA, imageName)
        cv2.imwrite(savePath, image)
        t1 = time.time()
        print('End! Case time: {}'.format(t1-t0))
    end = time.time()
    print('Finlly! Total time of build TrainB: {}'.format(end-start))
    '''