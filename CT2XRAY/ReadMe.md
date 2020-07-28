# Introduction
This is the method used to genereate synthetic Xray images from CT volumes, as introduced in the CVPR 2019 paper: [X2CT-GAN: Reconstructing CT from Biplanar X-Rays with Generative Adversarial Networks.](https://arxiv.org/abs/1905.06902)

As many researchers are interested in the method, we finally took sometime to organzie the source code and release them. Sorry for the delay and let us know if you have any questions. 

It includes three files:
- ctpro.py          CT pre-processing functions, mainly normalization and resampling methods.
- xraypro.py        DRR software script to generate synthetic Xray images from normalized CT volumes.
- ctxray_utils.py   Utility functions.

# Dependence

glob, scipy, numpy, cv2, pfm, matplotlib, pydicom, SiimpleITK


# Usage Guidance

1. Install the DRR software from [here](https://www.plastimatch.org/) on a Windows computer, and we use 1.7.3.12-win64 version in our original work.
2. In ctpro.py, first set data path;
    * root_path ： original CT volume root 
    * mask_root_path ： CT mask without table
    * save_root_path ： save dir 
    * then execute ctpro.py to generate the normalized CT volumes.
3. In xraypro.py, again set the data path;
    * root_path ： the normalized data path, same as the save_root_path in ctpro.py file
    * save_root_path ：the Xray output path
    * plasti_path ： DDR software executable file location 
    * then execute xraypro.py to generate the synthetic Xrays. 
4. Simple, right? If you feel our work is helpful to your project, please refer our CVPR work in your literature. 
5. And yes, it still not done. :) To have realistic looking Xrays, we did another CycleGAN trick for the style transfer. You can directly use the official release or waiting for our version, coming soon. 