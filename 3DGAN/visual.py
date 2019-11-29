# ------------------------------------------------------------------------------
# Copyright (c) Tencent
# Licensed under the GPLv3 License.
# Created by Kai Ma (makai0324@gmail.com)
# ------------------------------------------------------------------------------

import argparse
from lib.config.config import cfg_from_yaml, cfg, merge_dict_and_yaml, print_easy_dict
from lib.dataset.factory import get_dataset
from lib.model.factory import get_model
from lib.utils import html
from lib.utils.visualizer import tensor_back_to_unnormalization, save_images, tensor_back_to_unMinMax
#from lib.utils.metrics_np import MAE, MSE, Peak_Signal_to_Noise_Rate, Structural_Similarity, Cosine_Similarity
from lib.utils import ct as CT
import copy
import tqdm
import torch
import numpy as np
import os

def parse_args():
  parse = argparse.ArgumentParser(description='CTGAN')
  parse.add_argument('--data', type=str, default='', dest='data',
                     help='input data ')
  parse.add_argument('--tag', type=str, default='', dest='tag',
                     help='distinct from other try')
  parse.add_argument('--dataroot', type=str, default='', dest='dataroot',
                     help='input data root')
  parse.add_argument('--dataset', type=str, default='', dest='dataset',
                     help='Train or test or valid')
  parse.add_argument('--datasetfile', type=str, default='', dest='datasetfile',
                     help='Train or test or valid file path')
  parse.add_argument('--ymlpath', type=str, default=None, dest='ymlpath',
                     help='config have been modified')
  parse.add_argument('--gpu', type=str, default='0,1', dest='gpuid',
                     help='gpu is split by ,')
  parse.add_argument('--dataset_class', type=str, default='unalign', dest='dataset_class',
                     help='Dataset class should select from unalign /')
  parse.add_argument('--model_class', type=str, default='cyclegan', dest='model_class',
                     help='Model class should select from cyclegan / ')
  parse.add_argument('--check_point', type=str, default=None, dest='check_point',
                     help='which epoch to load? ')
  parse.add_argument('--latest', action='store_true', dest='latest',
                     help='set to latest to use latest cached model')
  parse.add_argument('--verbose', action='store_true', dest='verbose',
                     help='if specified, print more debugging information')
  parse.add_argument('--load_path', type=str, default=None, dest='load_path',
                     help='if load_path is not None, model will load from load_path')
  parse.add_argument('--how_many', type=int, dest='how_many', default=50,
                     help='if specified, print more debugging information')
  parse.add_argument('--resultdir', type=str, default='', dest='resultdir',
                     help='dir to save result')
  args = parse.parse_args()
  return args

if __name__ == '__main__':
  args = parse_args()

  # check gpu
  if args.gpuid == '':
    args.gpu_ids = []
  else:
    if torch.cuda.is_available():
      split_gpu = str(args.gpuid).split(',')
      args.gpu_ids = [int(i) for i in split_gpu]
    else:
      print('There is no gpu!')
      exit(0)

  # check point
  if args.check_point is None:
    args.epoch_count = 1
  else:
    args.epoch_count = int(args.check_point)

  # merge config with yaml
  if args.ymlpath is not None:
    cfg_from_yaml(args.ymlpath)
  # merge config with argparse
  opt = copy.deepcopy(cfg)
  opt = merge_dict_and_yaml(args.__dict__, opt)
  print_easy_dict(opt)

  opt.serial_batches = True

  # add data_augmentation
  datasetClass, _, dataTestClass, collateClass = get_dataset(opt.dataset_class)
  opt.data_augmentation = dataTestClass

  # get dataset
  dataset = datasetClass(opt)
  print('DataSet is {}'.format(dataset.name))
  dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=1,
    shuffle=False,
    num_workers=int(opt.nThreads),
    collate_fn=collateClass)

  dataset_size = len(dataloader)
  print('#Test images = %d' % dataset_size)

  # get model
  gan_model = get_model(opt.model_class)()
  print('Model --{}-- will be Used'.format(gan_model.name))

  # set to test
  gan_model.eval()

  gan_model.init_process(opt)
  total_steps, epoch_count = gan_model.setup(opt)

  # must set to test Mode again, due to  omission of assigning mode to network layers
  # model.training is test, but BN.training is training
  if opt.verbose:
    print('## Model Mode: {}'.format('Training' if gan_model.training else 'Testing'))
    for i, v in gan_model.named_modules():
      print(i, v.training)

  if 'batch' in opt.norm_G:
    gan_model.eval()
  elif 'instance' in opt.norm_G:
    gan_model.eval()
    # instance norm in training mode is better
    for name, m in gan_model.named_modules():
      if m.__class__.__name__.startswith('InstanceNorm'):
        m.train()
  else:
    raise NotImplementedError()

  if opt.verbose:
    print('## Change to Model Mode: {}'.format('Training' if gan_model.training else 'Testing'))
    for i, v in gan_model.named_modules():
      print(i, v.training)

  web_dir = os.path.join(opt.resultdir, opt.data, '%s_%s' % (opt.dataset, opt.check_point))
  webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.data, opt.dataset, opt.check_point))
  ctVisual = CT.CTVisual()

  avg_dict = dict()
  for epoch_i, data in tqdm.tqdm(enumerate(dataloader)):

    gan_model.set_input(data)
    gan_model.test()

    visuals = gan_model.get_current_visuals()
    img_path = gan_model.get_image_paths()

    if epoch_i <= opt.how_many:
      #
      # Image HTML
      #
      save_images(webpage, visuals, img_path, gan_model.get_normalization_list(), max_image=50)
      #
      # CT Source
      #
      generate_CT = visuals['G_fake'].data.clone().cpu().numpy()
      real_CT = visuals['G_real'].data.clone().cpu().numpy()
      # To NDHW
      if 'std' in opt.dataset_class or 'baseline' in opt.dataset_class:
        generate_CT_transpose = generate_CT
        real_CT_transpose = real_CT
      else:
        generate_CT_transpose = np.transpose(generate_CT, (0, 2, 1, 3))
        real_CT_transpose = np.transpose(real_CT, (0, 2, 1, 3))
      # Inveser Deepth
      generate_CT_transpose = generate_CT_transpose[:, ::-1, :, :]
      real_CT_transpose = real_CT_transpose[:, ::-1, :, :]
      # To [0, 1]
      generate_CT_transpose = tensor_back_to_unnormalization(generate_CT_transpose, opt.CT_MEAN_STD[0],
                                                             opt.CT_MEAN_STD[1])
      real_CT_transpose = tensor_back_to_unnormalization(real_CT_transpose, opt.CT_MEAN_STD[0], opt.CT_MEAN_STD[1])
      # Clip generate_CT
      generate_CT_transpose = np.clip(generate_CT_transpose, 0, 1)

      # #
      # # Evaluate Part
      # #
      # mae = MAE(real_CT_transpose, generate_CT_transpose, size_average=False)
      # mse = MSE(real_CT_transpose, generate_CT_transpose, size_average=False)
      # cosinesimilarity = Cosine_Similarity(real_CT_transpose, generate_CT_transpose, size_average=False)
      # psnr = Peak_Signal_to_Noise_Rate(real_CT_transpose, generate_CT_transpose, size_average=False, PIXEL_MAX=1.0)
      # ssim = Structural_Similarity(real_CT_transpose, generate_CT_transpose, size_average=False, PIXEL_MAX=1.0)
      #
      # metrics_list = [('MAE', mae), ('MSE', mse), ('CosineSimilarity', cosinesimilarity), ('PSNR-1', psnr[0]),
      #                 ('PSNR-2', psnr[1]), ('PSNR-3', psnr[2]), ('PSNR-avg', psnr[3]),
      #                 ('SSIM-1', ssim[0]), ('SSIM-2', ssim[1]), ('SSIM-3', ssim[2]), ('SSIM-avg', ssim[3])]

      # To HU coordinate
      generate_CT_transpose = tensor_back_to_unMinMax(generate_CT_transpose, opt.CT_MIN_MAX[0], opt.CT_MIN_MAX[1]).astype(np.int32) - 1024
      real_CT_transpose = tensor_back_to_unMinMax(real_CT_transpose, opt.CT_MIN_MAX[0], opt.CT_MIN_MAX[1]).astype(np.int32) - 1024
      # Save
      name1 = os.path.splitext(os.path.basename(img_path[0][0]))[0]
      name2 = os.path.split(os.path.dirname(img_path[0][0]))[-1]
      name = name2 + '_' + name1
      image_root = os.path.join(web_dir, 'CT', name)
      if not os.path.exists(image_root):
        os.makedirs(image_root)
      save_path = os.path.join(image_root, 'fake_ct.mha')
      ctVisual.save(generate_CT_transpose.squeeze(0), spacing=(1.0, 1.0, 1.0), origin=(0,0,0), path=save_path)
      save_path = os.path.join(image_root, 'real_ct.mha')
      ctVisual.save(real_CT_transpose.squeeze(0), spacing=(1.0, 1.0, 1.0), origin=(0, 0, 0), path=save_path)

    else:
      break
    del visuals, img_path
  webpage.save()

