# ------------------------------------------------------------------------------
# Copyright (c) Tencent
# Licensed under the GPLv3 License.
# Created by Kai Ma (makai0324@gmail.com)
# ------------------------------------------------------------------------------
import os
import time
import glob
import scipy
import numpy as np
from ctxray_utils import load_scan_mhda, save_scan_mhda, bedstriping

# resample to stantard spacing (keep physical scale stable, change pixel numbers)
def resample(image, spacing, new_spacing=[1,1,1]):
    # .mhd image order : z, y, x
    if not isinstance(spacing, np.ndarray):
        spacing = np.array(spacing)
    if not isinstance(new_spacing, np.ndarray):
        new_spacing = np.array(new_spacing)
    spacing = spacing[::-1]
    new_spacing = new_spacing[::-1]
    
    spacing = np.array(list(spacing))
    resize_factor = spacing / new_spacing
    new_real_shape = image.shape * resize_factor
    new_shape = np.round(new_real_shape)
    real_resize_factor = new_shape / image.shape
    new_spacing = spacing / real_resize_factor
    image = scipy.ndimage.interpolation.zoom(image, real_resize_factor, mode='nearest')
    return image, new_spacing

# crop to stantard shape (scale x scale x scale) (keep pixel scale consistency)
def crop_to_standard(scan, scale):
    z, y, x = scan.shape    
    if z >= scale:
        ret_scan = scan[z-scale:z, :, :]
    else:        
        temp1 = np.zeros(((scale-z)//2, y, x))
        temp2 = np.zeros(((scale-z)-(scale-z)//2, y, x))
        ret_scan = np.concatenate((temp1, scan, temp2), axis=0)
    z, y, x = ret_scan.shape
    if y >= scale:
        ret_scan = ret_scan[:, (y-scale)//2:(y+scale)//2, :]
    else:
        temp1 = np.zeros((z, (scale-y)//2, x))
        temp2 = np.zeros((z, (scale-y)-(scale-y)//2, x))
        ret_scan = np.concatenate((temp1, ret_scan, temp2), axis=1)
    z, y, x = ret_scan.shape
    if x >= scale:
        ret_scan = ret_scan[:, :, (x-scale)//2:(x+scale)//2]
    else:
        temp1 = np.zeros((z, y, (scale-x)//2))
        temp2 = np.zeros((z, y, (scale-x)-(scale-x)//2))
        ret_scan = np.concatenate((temp1, ret_scan, temp2), axis=2)
    return ret_scan
        

if __name__ == '__main__':
    root_path = 'F:/Data/LIDC-IDRI_Volumes/'
    mask_root_path = 'F:/Data/LIDC-IDRI_BodyMask/'
    save_root_path = 'D:/LIDC_TEST'
    files_list = glob.glob(root_path + '*.mhd')
    files_list = sorted(files_list)
    
    start = time.time()
    for fileIndex, filePath in enumerate(files_list):
        t0 = time.time()
        _, file = os.path.split(filePath)
        fileName = os.path.splitext(file)[0]
        print('Begin {}/{}: {}'.format(fileIndex+1, len(files_list), fileName))
        maskFile = fileName + '_outMultiLabelMask' + '.mhd'
        maskPath = os.path.join(mask_root_path, maskFile)
        saveDir = os.path.join(save_root_path, fileName)
        if not os.path.exists(saveDir):
            os.makedirs(saveDir)
        savePath = os.path.join(saveDir, '{}.mha'.format(fileName))
        ct_itk, ct_scan, ori_origin, ori_size, ori_spacing= load_scan_mhda(filePath)
        print("Old : ", ori_size)
        _, ct_mask, _, _, _ = load_scan_mhda(maskPath)
        
        bedstrip_scan = bedstriping(ct_scan, ct_mask)
        new_scan, new_spacing = resample(bedstrip_scan, ori_spacing, [1, 1, 1])
        
        print("Std : ", new_scan.shape[::-1])
        std_scan = crop_to_standard(new_scan, scale=320)
        
        save_scan_mhda(std_scan, (0, 0, 0), new_spacing, savePath)
        _, _, _, new_size, new_spacing = load_scan_mhda(savePath)
        print("New : ", new_size)
        t1 = time.time()
        print('End! Case time: {}'.format(t1-t0))
    end = time.time()
    print('Finally! Total time: {}'.format(end-start))
