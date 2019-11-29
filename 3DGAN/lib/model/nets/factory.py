# ------------------------------------------------------------------------------
# Copyright (c) Tencent
# Licensed under the GPLv3 License.
# Created by Kai Ma (makai0324@gmail.com)
# ------------------------------------------------------------------------------

from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

from .patch_discriminator import NLayer_3D_Discriminator, NLayer_2D_Discriminator, Multiscale_3D_Discriminator, Multiscale_2D_Discriminator
from .utils import *


def define_D(input_nc, ndf, which_model_netD, n_layers_D=3,
             norm='batch', use_sigmoid=False, init_type='normal',
             gpu_ids=[], getIntermFeat=False, num_D=3, n_out_channels=1):
  netD = None
  norm_layer = get_norm_layer(norm_type=norm)

  if which_model_netD == 'basic3d':
    netD = NLayer_3D_Discriminator(input_nc, ndf, n_layers=n_layers_D,
                                   norm_layer=norm_layer, use_sigmoid=use_sigmoid,
                                   getIntermFeat=getIntermFeat, n_out_channels=n_out_channels)
  elif which_model_netD == 'basic2d':
    netD = NLayer_2D_Discriminator(input_nc, ndf, n_layers=n_layers_D,
                                   norm_layer=norm_layer, use_sigmoid=use_sigmoid,
                                   getIntermFeat=getIntermFeat, n_out_channels=n_out_channels)
  elif which_model_netD == 'multi3d':
    netD = Multiscale_3D_Discriminator(input_nc, ndf, n_layers=n_layers_D,
                                   norm_layer=norm_layer, use_sigmoid=use_sigmoid,
                                   getIntermFeat=getIntermFeat, num_D=num_D, n_out_channels=n_out_channels)
  elif which_model_netD == 'multi2d':
    netD = Multiscale_2D_Discriminator(input_nc, ndf, n_layers=n_layers_D,
                                   norm_layer=norm_layer, use_sigmoid=use_sigmoid,
                                   getIntermFeat=getIntermFeat, num_D=num_D, n_out_channels=n_out_channels)
  else:
    raise NotImplementedError('Discriminator model name [%s] is not recognized' %
                              which_model_netD)
  return init_net(netD, init_type, gpu_ids)


def define_3DG(noise_len, input_shape, output_shape, input_nc, output_nc, ngf, which_model_netG, n_downsampling,
             norm='batch', use_dropout=False, init_type='normal', gpu_ids=[], n_blocks=9,
               encoder_input_shape=(128,128), encoder_input_nc=1, encoder_norm='instance2d', encoder_blocks=3, skip_num=1, activation_type='relu', opt=None):
  netG = None
  decoder_norm_layer = get_norm_layer(norm_type=norm)
  encoder_norm_layer = get_norm_layer(norm_type=encoder_norm)
  activation_layer = get_generator_activation_func(opt, activation_type)

  if which_model_netG == 'singleview_network_denseUNet_transposed':
    from .generator.dense_generator_multiview import UNetLike_DownStep5
    netG = UNetLike_DownStep5(input_shape=encoder_input_shape[0], encoder_input_channels=encoder_input_nc, decoder_output_channels=output_nc, decoder_out_activation=activation_layer, encoder_norm_layer=encoder_norm_layer, decoder_norm_layer=decoder_norm_layer, upsample_mode='transposed')
  elif which_model_netG == 'singleview_network_denseUNet_transposed_withoutskip':
    from .generator.dense_generator_withoutskip_multiview import UNetLike_DownStep5
    netG = UNetLike_DownStep5(input_shape=encoder_input_shape[0], encoder_input_channels=encoder_input_nc, decoder_output_channels=output_nc, decoder_out_activation=activation_layer, encoder_norm_layer=encoder_norm_layer, decoder_norm_layer=decoder_norm_layer, upsample_mode='transposed')

  elif which_model_netG == 'multiview_network_denseUNetFuse_transposed_withoutskip':
    from .generator.dense_generator_withoutskip_multiview import UNetLike_DownStep5, MultiView_UNetLike_DenseDimensionNet
    netG = MultiView_UNetLike_DenseDimensionNet(view1Model=UNetLike_DownStep5(input_shape=encoder_input_shape[0], encoder_input_channels=encoder_input_nc, decoder_output_channels=output_nc, decoder_out_activation=activation_layer, encoder_norm_layer=encoder_norm_layer, decoder_norm_layer=decoder_norm_layer, upsample_mode='transposed', decoder_feature_out=True), view2Model=UNetLike_DownStep5(input_shape=encoder_input_shape[0], encoder_input_channels=encoder_input_nc, decoder_output_channels=output_nc, decoder_out_activation=activation_layer, encoder_norm_layer=encoder_norm_layer, decoder_norm_layer=decoder_norm_layer, upsample_mode='transposed', decoder_feature_out=True), view1Order=opt.CTOrder_Xray1, view2Order=opt.CTOrder_Xray2, backToSub=True, decoder_output_channels=output_nc, decoder_out_activation=activation_layer, decoder_block_list=[1, 1, 1, 1, 1, 0], decoder_norm_layer=decoder_norm_layer, upsample_mode='transposed')

  elif which_model_netG == 'multiview_network_denseUNetFuse_transposed_withoutconnect':
    from .generator.dense_generator_multiview_withoutc import UNetLike_DownStep5, MultiView_UNetLike_DenseDimensionNet
    netG = MultiView_UNetLike_DenseDimensionNet(view1Model=UNetLike_DownStep5(input_shape=encoder_input_shape[0], encoder_input_channels=encoder_input_nc, decoder_output_channels=output_nc, decoder_out_activation=activation_layer, encoder_norm_layer=encoder_norm_layer, decoder_norm_layer=decoder_norm_layer, upsample_mode='transposed', decoder_feature_out=True), view2Model=UNetLike_DownStep5(input_shape=encoder_input_shape[0], encoder_input_channels=encoder_input_nc, decoder_output_channels=output_nc, decoder_out_activation=activation_layer, encoder_norm_layer=encoder_norm_layer, decoder_norm_layer=decoder_norm_layer, upsample_mode='transposed', decoder_feature_out=True), view1Order=opt.CTOrder_Xray1, view2Order=opt.CTOrder_Xray2, backToSub=True, decoder_output_channels=output_nc, decoder_out_activation=activation_layer, decoder_block_list=[1, 1, 1, 1, 1, 0], decoder_norm_layer=decoder_norm_layer, upsample_mode='transposed')

  elif which_model_netG == 'multiview_network_denseUNetFuse_transposed_withoutconnectAndskip':
    from .generator.dense_generator_multiview_withoutc_skip import UNetLike_DownStep5, MultiView_UNetLike_DenseDimensionNet
    netG = MultiView_UNetLike_DenseDimensionNet(view1Model=UNetLike_DownStep5(input_shape=encoder_input_shape[0], encoder_input_channels=encoder_input_nc, decoder_output_channels=output_nc, decoder_out_activation=activation_layer, encoder_norm_layer=encoder_norm_layer, decoder_norm_layer=decoder_norm_layer, upsample_mode='transposed', decoder_feature_out=True), view2Model=UNetLike_DownStep5(input_shape=encoder_input_shape[0], encoder_input_channels=encoder_input_nc, decoder_output_channels=output_nc, decoder_out_activation=activation_layer, encoder_norm_layer=encoder_norm_layer, decoder_norm_layer=decoder_norm_layer, upsample_mode='transposed', decoder_feature_out=True), view1Order=opt.CTOrder_Xray1, view2Order=opt.CTOrder_Xray2, backToSub=True, decoder_output_channels=output_nc, decoder_out_activation=activation_layer, decoder_block_list=[1, 1, 1, 1, 1, 0], decoder_norm_layer=decoder_norm_layer, upsample_mode='transposed')

  elif which_model_netG == 'multiview_network_denseUNetFuse_transposed':
    from .generator.dense_generator_multiview import UNetLike_DownStep5, MultiView_UNetLike_DenseDimensionNet
    netG = MultiView_UNetLike_DenseDimensionNet(view1Model=UNetLike_DownStep5(input_shape=encoder_input_shape[0], encoder_input_channels=encoder_input_nc, decoder_output_channels=output_nc, decoder_out_activation=activation_layer, encoder_norm_layer=encoder_norm_layer, decoder_norm_layer=decoder_norm_layer, upsample_mode='transposed', decoder_feature_out=True), view2Model=UNetLike_DownStep5(input_shape=encoder_input_shape[0], encoder_input_channels=encoder_input_nc, decoder_output_channels=output_nc, decoder_out_activation=activation_layer, encoder_norm_layer=encoder_norm_layer, decoder_norm_layer=decoder_norm_layer, upsample_mode='transposed', decoder_feature_out=True), view1Order=opt.CTOrder_Xray1, view2Order=opt.CTOrder_Xray2, backToSub=True, decoder_output_channels=output_nc, decoder_out_activation=activation_layer, decoder_block_list=[1, 1, 1, 1, 1, 0], decoder_norm_layer=decoder_norm_layer, upsample_mode='transposed')

  elif which_model_netG == 'multiview_network_denseUNetNoFuse_transposed':
    from .generator.dense_generator_multiview import UNetLike_DownStep5, MultiView_UNetLike_DenseDimensionNet
    netG = MultiView_UNetLike_DenseDimensionNet(view1Model=UNetLike_DownStep5(input_shape=encoder_input_shape[0], encoder_input_channels=encoder_input_nc, decoder_output_channels=output_nc, decoder_out_activation=activation_layer, encoder_norm_layer=encoder_norm_layer, decoder_norm_layer=decoder_norm_layer, upsample_mode='transposed', decoder_feature_out=True), view2Model=UNetLike_DownStep5(input_shape=encoder_input_shape[0], encoder_input_channels=encoder_input_nc, decoder_output_channels=output_nc, decoder_out_activation=activation_layer, encoder_norm_layer=encoder_norm_layer, decoder_norm_layer=decoder_norm_layer, upsample_mode='transposed', decoder_feature_out=True), view1Order=opt.CTOrder_Xray1, view2Order=opt.CTOrder_Xray2, backToSub=False, decoder_output_channels=output_nc, decoder_out_activation=activation_layer, decoder_block_list=[1, 1, 1, 1, 1, 0], decoder_norm_layer=decoder_norm_layer, upsample_mode='transposed')

  elif which_model_netG == 'singleview_2D':
    from .singleXray3DVolumes import Xray3DVolumes_2DModel
    netG = Xray3DVolumes_2DModel(activation_layer)

  else:
    raise NotImplementedError('Generator model name [%s] is not recognized' % which_model_netG)

  return init_net(netG, init_type, gpu_ids)