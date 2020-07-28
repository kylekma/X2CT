# ------------------------------------------------------------------------------
# Copyright (c) Tencent
# Licensed under the GPLv3 License.
# Created by Kai Ma (makai0324@gmail.com)
# ------------------------------------------------------------------------------
import os
import cv2
import pfm
import time
import numpy as np
import matplotlib.pyplot as plt
from subprocess import check_output as qx
from ctxray_utils import load_scan_mhda, save_scan_mhda


# compute xray source center in world coordinate
def get_center(origin, size, spacing):
    origin = np.array(origin)
    size = np.array(size)
    spacing = np.array(spacing)
    center = origin + (size - 1) / 2 * spacing
    return center

# convert a ndarray to string
def array2string(ndarray):
    ret = ""
    for i in ndarray:
        ret = ret + str(i) + " "
    return ret[:-2]

# save a .pfm file as a .png file
def savepng(filename, direction):
    raw_data , scale = pfm.read(filename)
    max_value = raw_data.max()
    im = (raw_data / max_value * 255).astype(np.uint8)
    # PA view should do additional left-right flip
    if direction == 1:
        im = np.fliplr(im)
    
    savedir, _ = os.path.split(filename)
    outfile = os.path.join(savedir, "xray{}.png".format(direction))
    # plt.imshow(im, cmap=plt.cm.gray)
    plt.imsave(outfile, im, cmap=plt.cm.gray)
    # plt.imsave saves an image with 32bit per pixel, but we only need one channel
    image = cv2.imread(outfile)
    gray = cv2.split(image)[0]
    cv2.imwrite(outfile, gray)


if __name__ == '__main__':
    root_path = 'D:/LIDC_TEST/'
    save_root_path = 'D:/LIDC_TEST'
    plasti_path = 'D:/Program Files/Plastimatch/bin'
    
    files_list = os.listdir(root_path)
    start = time.time()
    for fileIndex, fileName in enumerate(files_list):
        t0 = time.time()
        print('Begin {}/{}: {}'.format(fileIndex+1, len(files_list), fileName))
        saveDir = os.path.join(root_path, fileName)
        if not os.path.exists(saveDir):
            os.makedirs(saveDir)
        # savePath is the .mha file store location
        savePath = os.path.join(saveDir, '{}.mha'.format(fileName))
        ct_itk, ct_scan, ori_origin, ori_size, ori_spacing = load_scan_mhda(savePath)
        # compute isocenter
        center = get_center(ori_origin, ori_size, ori_spacing)
        # map the .mha file value to (-1000, 1000)
        cmd_str = plasti_path + '/plastimatch adjust --input {} --output {}\
                --pw-linear "0, -1000"'.format(savePath, saveDir+'/out.mha')  
        output = qx(cmd_str)
        # get virtual xray file
        directions = [1, 2]
        for i in directions:
            if i == 1:
                nrm = "0 1 0"
            else:
                nrm = "1 0 0"    
            '''
            plastimatch usage
            -t : save format
            -g : sid sad [DistanceSourceToPatient]:541 
                         [DistanceSourceToDetector]:949.075012
            -r : output image resolution
            -o : isocenter position
            -z : physical size of imager
            -I : input file in .mha format
            -O : output prefix
            '''
            cmd_str = plasti_path + '/drr -t pfm -nrm "{}" -g "541 949" \
                    -r "320 320" -o "{}" -z "500 500" -I {} -O {}'.format\
                    (nrm, array2string(center), saveDir+'/out.mha', saveDir+'/{}'.format(i))
            output = qx(cmd_str)
            # plastimatch would add a 0000 suffix 
            pfmFile = saveDir+'/{}'.format(i) + '0000.pfm'
            savepng(pfmFile, i)
        # remove the temp .mha file couse it is so large
        os.remove(saveDir+'/out.mha')
        t1 = time.time()
        print('End! Case time: {}'.format(t1-t0))
    end = time.time()
    print('Finally! Total time: {}'.format(end-start))
