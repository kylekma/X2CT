# ------------------------------------------------------------------------------
# Copyright (c) Tencent
# Licensed under the GPLv3 License.
# Created by Kai Ma (makai0324@gmail.com)
# ------------------------------------------------------------------------------

from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import torch
import torch.nn as nn
import os
from .factory import get_scheduler
from collections import OrderedDict

class Base_Model(nn.Module):
  '''
  Base Model
  Used to be inherited
  '''
  def __init__(self):
    super(Base_Model, self).__init__()

  @property
  def name(self):
    return 'BaseModel'

  '''
  Init network architecture
  '''
  def init_network(self, opt):
    self.opt = opt
    self.gpu_ids = opt.gpu_ids
    self.device = torch.device('cuda:{}'.format(self.gpu_ids[0])) \
      if (self.gpu_ids) else torch.device('cpu')
    self.save_root = os.path.join(opt.MODEL_SAVE_PATH, self.name, opt.data, opt.tag)
    self.save_dir = os.path.join(self.save_root, 'checkpoint')
    if not os.path.exists(self.save_dir):
      os.makedirs(self.save_dir)
    self.loss_names = []
    self.metrics_names = []
    self.model_names = []
    self.visual_names = []
    self.image_paths = []
    self.optimizers = []

  def init_loss(self, opt):
    pass

  def init_process(self, opt):
    self.init_network(opt)
    if self.training:
      self.init_loss(opt)

  '''
  Train -Forward and Backward
  '''
  # used in test time, wrapping `forward` in no_grad() so we don't save
  # intermediate steps for backprop
  def test(self):
    with torch.no_grad():
      self.forward()

  def set_input(self, input):
    self.input = input

  def optimize_parameters(self):
    pass

  # update learning rate (called once every epoch)
  def update_learning_rate(self, total_step):
    for scheduler in self.schedulers:
      scheduler.step(total_step)
    lr = self.optimizers[0].param_groups[0]['lr']
    print('learning rate = %.7f' % lr)

  # set requies_grad=Fasle to avoid computation
  def set_requires_grad(self, nets, requires_grad=False):
    if not isinstance(nets, list):
      nets = [nets]
    for net in nets:
      if net is not None:
        for param in net.parameters():
          param.requires_grad = requires_grad
  '''
  Visualize
  '''
  # get metrics
  def get_current_metrics(self):
    errors_ret = OrderedDict()
    for name in self.metrics_names:
      if isinstance(name, str):
        # float(...) works for both scalar tensor and float number
        errors_ret[name] = float(getattr(self, 'metrics_' + name).cpu())
    return errors_ret

  # return traning losses/errors. train.py will print out these errors as debugging information
  def get_current_losses(self):
    errors_ret = OrderedDict()
    for name in self.loss_names:
      if isinstance(name, str):
        # float(...) works for both scalar tensor and float number
        errors_ret[name] = float(getattr(self, 'loss_' + name).cpu())
    return errors_ret

  # return visualization images. train.py will display these images, and save the images to a html
  def get_current_visuals(self):
    visual_ret = OrderedDict()
    for name in self.visual_names:
      if isinstance(name, str):
        visual_ret[name] = getattr(self, name)
    return visual_ret

  # get image paths
  def get_image_paths(self):
    return self.image_paths

  # get norm
  def get_normalization_list(self):
    pass

  '''
  Other visual and save model functions
  '''
  # load and print networks; create schedulers
  def setup(self, opt, parser=None):
    total_steps = 0
    epoch_count = 0
    if self.training:
      self.schedulers = [get_scheduler(optimizer, opt) for optimizer in self.optimizers]

    if not self.training or (opt.check_point is not None) or (opt.load_path is not None):
      total_steps, epoch_count = self.load_networks(opt.check_point, opt.load_path, opt.latest)

    self.print_networks(opt.verbose)
    return total_steps, epoch_count

  # print network information
  def print_networks(self, verbose):
    print('---------- Networks initialized -------------')
    for name in self.model_names:
      if isinstance(name, str):
        net = getattr(self, 'net' + name)
        num_params = 0
        for param in net.parameters():
          num_params += param.numel()
        if verbose:
          print(net)
        print('[Network %s] Total number of parameters : %.3f M' % (name, num_params / 1e6))
    print('-----------------------------------------------')

  # save models to the disk
  def save_networks(self, which_epoch, total_steps, latest=False):
    for name in self.model_names:
      if isinstance(name, str):
        if latest:
          save_filename = '{}_net_{}.pth'.format('latest', name)
          save_path = os.path.join(self.save_dir, save_filename)
        else:
          save_filename = '{}_net_{}.pth'.format(which_epoch, name)
          save_path = os.path.join(self.save_dir, str(which_epoch))
          if not os.path.exists(save_path):
            os.makedirs(save_path)
          save_path = os.path.join(self.save_dir, str(which_epoch), save_filename)

        net = getattr(self, 'net' + name)

        if len(self.gpu_ids) > 0 and torch.cuda.is_available():
          param_dict = net.module.cpu().state_dict()
          net.cuda(self.device)
        else:
          param_dict = net.cpu().state_dict()

        save_dict = {
          'iters': total_steps,
          'epoch': which_epoch,
          'state_dict': param_dict,
          'lr': self.optimizers[0].param_groups[0]['lr']
        }
        torch.save(save_dict, save_path)

  # load models from the disk
  def load_networks(self, which_epoch, load_Path=None, latest=False):
    total_steps = 0
    epoch_count = 0
    for name in self.model_names:
      if isinstance(name, str):
        if latest:
          load_filename = '{}_net_{}.pth'.format('latest', name)
          if load_Path is not None:
            load_path = os.path.join(load_Path, load_filename)
          else:
            load_path = os.path.join(self.save_dir, load_filename)
        else:
          load_filename = '{}_net_{}.pth'.format(which_epoch, name)
          if load_Path is not None:
            load_path = os.path.join(load_Path, str(which_epoch), load_filename)
          else:
            load_path = os.path.join(self.save_dir, str(which_epoch), load_filename)
        net = getattr(self, 'net' + name)
        if isinstance(net, torch.nn.DataParallel):
          net = net.module
        print('**loading the model from %s' % load_path)
        # if you are using PyTorch newer than 0.4 (e.g., built from
        # GitHub source), you can remove str() on self.device
        # save_dict = torch.load(load_path, map_location=str(self.device))
        save_dict = torch.load(load_path)
        state_dict = save_dict['state_dict']

        # patch InstanceNorm checkpoints prior to 0.4
        for key in list(state_dict.keys()):  # need to copy keys here because we mutate in loop
          self.__patch_instance_norm_state_dict(state_dict, net, key.split('.'))

        will_state_dict = {i:j for i,j in state_dict.items() if i in net.state_dict().keys()}
        print('**loading {} parameters from {}(saved model), net size = {}'.format(len(will_state_dict), len(state_dict), len(net.state_dict())))

        net.load_state_dict(will_state_dict)
        net.to(self.device)
        total_steps = save_dict['iters']
        epoch_count = save_dict['epoch']
    return total_steps, epoch_count


  def __patch_instance_norm_state_dict(self, state_dict, module, keys, i=0):
    key = keys[i]
    if i + 1 == len(keys):  # at the end, pointing to a parameter/buffer
      if module.__class__.__name__.startswith('InstanceNorm') and \
          (key == 'running_mean' or key == 'running_var'):
        if getattr(module, key) is None:
          print('wrong!!'*10)
          state_dict.pop('.'.join(keys))
    else:
      self.__patch_instance_norm_state_dict(state_dict, getattr(module, key), keys, i + 1)