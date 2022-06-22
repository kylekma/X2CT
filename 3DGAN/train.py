# ------------------------------------------------------------------------------
# Copyright (c) Tencent
# Licensed under the GPLv3 License.
# Created by Kai Ma (makai0324@gmail.com)
# ------------------------------------------------------------------------------
import os
from sre_constants import JUMP

from tabnanny import verbose 
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import argparse
from lib.config.config import cfg_from_yaml, cfg, merge_dict_and_yaml, print_easy_dict
from lib.dataset.factory import get_dataset
from lib.model.factory import get_model
from lib.model.multiView_AutoEncoder import ResUNet
from lib.model.multiView_AutoEncoder import ResUNet_Down
from lib.model.multiView_AutoEncoder import ResUNet_up
import copy
import torch
import time
import torch.optim as optim
from tqdm import tqdm
#from logger import Logger
#import sys

"""
def batch_learn(batch):

  return 

"""

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
  parse.add_argument('--valid_dataset', type=str, default=None, dest='valid_dataset',
                     help='Train or test or valid')
  parse.add_argument('--datasetfile', type=str, default='', dest='datasetfile',
                     help='Train or test or valid file path')
  parse.add_argument('--valid_datasetfile', type=str, default='', dest='valid_datasetfile',
                     help='Train or test or valid file path')
  parse.add_argument('--ymlpath', type=str, default=None, dest='ymlpath',
                     help='config have been modified')
  parse.add_argument('--gpu', type=str, default='0,1', dest='gpuid',
                     help='gpu is split by ,')
  parse.add_argument('--dataset_class', type=str, default='align', dest='dataset_class',
                     help='Dataset class should select from align /')
  parse.add_argument('--model_class', type=str, default='simpleGan', dest='model_class',
                     help='Model class should select from simpleGan / ')
  parse.add_argument('--check_point', type=str, default=None, dest='check_point',
                     help='which epoch to load? ')
  parse.add_argument('--load_path', type=str, default=None, dest='load_path',
                     help='if load_path is not None, model will load from load_path')
  parse.add_argument('--latest', action='store_true', dest='latest',
                     help='set to latest to use latest cached model')
  parse.add_argument('--verbose', action='store_true', dest='verbose',
                     help='if specified, print more debugging information')
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
    args.epoch_count = int(args.check_point) + 1

  # merge config with yaml
  if args.ymlpath is not None:
    cfg_from_yaml(args.ymlpath)
  # merge config with argparse
  opt = copy.deepcopy(cfg)
  opt = merge_dict_and_yaml(args.__dict__, opt)
  print_easy_dict(opt)

  #torch.cuda.empty_cache()

  # add data_augmentation
  datasetClass, augmentationClass, dataTestClass, collateClass = get_dataset(opt.dataset_class)
  opt.data_augmentation = augmentationClass

  # valid dataset
  if args.valid_dataset is not None:
    valid_opt = copy.deepcopy(opt)
    valid_opt.data_augmentation = dataTestClass
   
    valid_opt.datasetfile = opt.valid_datasetfile


    valid_dataset = datasetClass(valid_opt)
    print('Valid DataSet is {}'.format(valid_dataset.name))
    valid_dataloader = torch.utils.data.DataLoader(
      valid_dataset,
      batch_size=1,
      shuffle=False,
      num_workers=int(valid_opt.nThreads),
      collate_fn=collateClass)
    valid_dataset_size = len(valid_dataloader)
    print('#validation images = %d' % valid_dataset_size)
  else:
    valid_dataloader = None

  # get dataset

  dataset = datasetClass(opt)
  print('DataSet is {}'.format(dataset.name))
  dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=opt.batch_size,
    shuffle=True,
    num_workers=int(opt.nThreads),
    collate_fn=collateClass)


  dataset_size = len(dataloader)
  print('#training images = %d' % dataset_size)

  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  autoencoder = ResUNet(in_channel=1,out_channel=1,training=True).to(device)
  auto_down = ResUNet_Down(in_channel = 1, out_channel=256).to(device)
  
  #print(autoencoder)
  #print(auto_down)
  

  autoencoder.train()

  # set to train
 

  pretrain_auto = {}
  pretrain_auto["alpha"] = 0.4
  pretrain_auto["epoch"] = 3
  #first batch size was 30
  pretrain_auto["batch"] = 35
  pretrain_auto["loss"] = torch.nn.L1Loss().to(device)
  #Gamma for next three items are 0.9
  #0.0001 err 0.0744 epoch 0 batch 30
  #0.00001 err 0.06 ep 0 batch 30 0.05-0.07
  #lr = 0.0005 batch 35 err 0.043-0.05
  #  
  pretrain_auto["optimizer"] = optim.Adam(autoencoder.parameters(),lr=0.00005)

  pretrain_auto["scheduler"] = optim.lr_scheduler.ExponentialLR(pretrain_auto["optimizer"],gamma=0.4, verbose=True)
  #pretrain_auto["scheduler"] = optim.lr_scheduler.LamdaLR(pretrain_auto["optimizer"],batch_learn)



  dataloader_auto = torch.utils.data.DataLoader(
    dataset,
    batch_size= 1,
    shuffle=True,
    num_workers=int(opt.nThreads),
    collate_fn=collateClass)

  predicts = None
  #remove below line when not debugging
  pretrain_auto["epoch"] = 3
  template = None
  
  
  #pretraining of autoencoder
  for epoch in range(pretrain_auto["epoch"]):
      correct = 0
      for i, data in enumerate(dataloader_auto):
        X = data[0]
        if template == None:
          template = data[3][0]
        #print(X)
        X = torch.unsqueeze(X,1)
        #print("fieeeeeem" + str(torch.sum(X)))
        X = X.to(device)

        pretrain_auto["optimizer"].zero_grad()
        
        predicts = autoencoder(X)

        loss0 = pretrain_auto["loss"](predicts[0],X)
        loss1 = pretrain_auto["loss"](predicts[1],X)
        loss2 = pretrain_auto["loss"](predicts[2],X)
        loss3 = pretrain_auto["loss"](predicts[3],X)
        #print("\n loss0: {}, loss1: {}, loss2: {}, loss3: {} \n".format(loss0.item(),loss1.item(),loss2.item(),loss3.item()))
        loss = loss3 + pretrain_auto["alpha"] *(loss0 + loss1 + loss2)
        loss.backward()
        pretrain_auto["optimizer"].step()
        if i % pretrain_auto["batch"] == 0:
          print("\n Epoch: {}, Loss: {}, Batch {}\n ".format(epoch, loss.item(),i))
          break
      pretrain_auto["scheduler"].step()


  
  template = template.to(device)

  template = torch.unsqueeze(template,dim=0)
  template = torch.unsqueeze(template,dim=0)
  keys = set(auto_down.state_dict().keys())
  auto_down.load_state_dict({k:v for k,v in autoencoder.state_dict().items() if k in keys})
  feature_map = auto_down(template)
  dim = feature_map.size()[2]
  #print(feature_map.size())
  #print(dim)
  
  auto_up = ResUNet_up(in_channel=256,out_channel=1,dim= dim).to(device)

  output = auto_up(feature_map)





  
  #print(feature_map)
  #print("SUM of feature map: {}".format(torch.sum(feature_map)))
  #exit() HERE

  
  
 

  # get model
  gan_model = get_model(opt.model_class)()
  print('Model --{}-- will be Used'.format(gan_model.name))
  gan_model.init_process(opt)
  total_steps, epoch_count = gan_model.setup(opt)
  gan_model.train()


  



  # visualizer
  from lib.utils.visualizer import Visualizer
  visualizer = Visualizer(log_dir=os.path.join(gan_model.save_root, 'train_log'))

  total_steps = total_steps

  # train discriminator more
  dataloader_iter_for_discriminator = iter(dataloader)


  #we need to add inout to optimize_paramets with output of autoencoder 
  # OLD train NOW finetune 
  for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):
    epoch_start_time = time.time()
    iter_data_time = time.time()

    for epoch_i, data in enumerate(dataloader):
      iter_start_time = time.time()

      total_steps += 1

      gan_model.set_input(data)
      t0 = time.time()
      gan_model.optimize_parameters()
      t1 = time.time()

      # if total_steps == 1:
      #   visualizer.add_graph(model=gan_model, input=gan_model.forward())

      # # visual gradient
      # if opt.verbose and total_steps % opt.print_freq == 0:
      #   for name, para in gan_model.named_parameters():
      #     visualizer.add_histogram('Grad_' + name, para.grad.data.clone().cpu().numpy(), step=total_steps)
      #     visualizer.add_histogram('Weight_' + name, para.data.clone().cpu().numpy(), step=total_steps)
      #   for name in gan_model.model_names:
      #     net = getattr(gan_model, 'net' + name)
      #     if hasattr(net, 'output_dict'):
      #       for name, out in net.output_dict.items():
      #         visualizer.add_histogram(name, out.numpy(), step=total_steps)

      # loss
      loss_dict = gan_model.get_current_losses()
      # visualizer.add_scalars('Train_Loss', loss_dict, step=total_steps)
      total_loss = visualizer.add_total_scalar('Total loss', loss_dict, step=total_steps)
      # visualizer.add_average_scalers('Epoch Loss', loss_dict, step=total_steps, write=False)
      # visualizer.add_average_scalar('Epoch total Loss', total_loss)

      # metrics
      # metrics_dict = gan_model.get_current_metrics()
      # visualizer.add_scalars('Train_Metrics', metrics_dict, step=total_steps)
      # visualizer.add_average_scalers('Epoch Metrics', metrics_dict, step=total_steps, write=False)

      if total_steps % opt.print_freq == 0:
        print('total step: {} timer: {:.4f} sec.'.format(total_steps, t1 - t0))
        print('epoch {}/{}, step{}:{} || total loss:{:.4f}'.format(epoch, opt.niter + opt.niter_decay,
                                                                   epoch_i, dataset_size, total_loss))
        print('||'.join(['{}: {:.4f}'.format(k, v) for k, v in loss_dict.items()]))
        # print('||'.join(['{}: {:.4f}'.format(k, v) for k, v in metrics_dict.items()]))
        print('')

      # if total_steps % opt.print_img_freq == 0:
      #   visualizer.add_image('Image', gan_model.get_current_visuals(), gan_model.get_normalization_list(), total_steps)

      '''
      WGAN
      '''
      if (opt.critic_times - 1) > 0:
        for critic_i in range(opt.critic_times - 1):
          try:
            data = next(dataloader_iter_for_discriminator)
            gan_model.set_input(data)
            gan_model.optimize_D()
          except:
            dataloader_iter_for_discriminator = iter(dataloader)
      del(loss_dict)

    # # save model every epoch
    # print('saving the latest model (epoch %d, total_steps %d)' %
    #       (epoch, total_steps))
    # gan_model.save_networks(epoch, total_steps, True)

    # save model several epoch
    if epoch % opt.save_epoch_freq == 0 and epoch >= opt.begin_save_epoch:
      print('saving the model at the end of epoch %d, iters %d' %
            (epoch, total_steps))
      gan_model.save_networks(epoch, total_steps)

    print('End of epoch %d / %d \t Time Taken: %d sec' %
          (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))
    ##########
    # For speed
    ##########
    # visualizer.add_image('Image_Epoch', gan_model.get_current_visuals(), gan_model.get_normalization_list(), epoch)
    # visualizer.add_average_scalers('Epoch Loss', None, step=epoch, write=True)
    # visualizer.add_average_scalar('Epoch total Loss', None, step=epoch, write=True)

    # visualizer.add_average_scalers('Epoch Metrics', None, step=epoch, write=True)

    # visualizer.add_scalar('Learning rate', gan_model.optimizers[0].param_groups[0]['lr'], epoch)
    gan_model.update_learning_rate(epoch)

    # # Test
    # if args.valid_dataset is not None:
    #   if epoch % opt.save_epoch_freq == 0 or epoch==1:
    #     gan_model.eval()
    #     iter_valid_dataloader = iter(valid_dataloader)
    #     for v_i in range(len(valid_dataloader)):
    #       data = next(iter_valid_dataloader)
    #       gan_model.set_input(data)
    #       gan_model.test()
    #
    #       if v_i < opt.howmany_in_train:
    #         visualizer.add_image('Test_Image', gan_model.get_current_visuals(), gan_model.get_normalization_list(), epoch*10+v_i, max_image=25)
    #
    #       # metrics
    #       metrics_dict = gan_model.get_current_metrics()
    #       visualizer.add_average_scalers('Epoch Test_Metrics', metrics_dict, step=total_steps, write=False)
    #     visualizer.add_average_scalers('Epoch Test_Metrics', None, step=epoch, write=True)
    #
    #     gan_model.train()