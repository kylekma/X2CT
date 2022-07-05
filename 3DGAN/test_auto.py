
import os
from sre_constants import JUMP

from tabnanny import verbose
from xmlrpc.client import Boolean 
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import argparse
from lib.config.config import cfg_from_yaml, cfg, merge_dict_and_yaml, print_easy_dict
from lib.dataset.factory import get_dataset
from lib.model.factory import get_model
from lib.model.multiView_AutoEncoder import ResUNet
from lib.model.multiView_AutoEncoder import ResUNet_Down
from lib.model.multiView_AutoEncoder import ResUNet_up
from lib.utils.visualizer import tensor_back_to_unnormalization, tensor_back_to_unMinMax
from lib.utils.metrics_np import MAE, MSE, Peak_Signal_to_Noise_Rate, Structural_Similarity, Cosine_Similarity, \
  Peak_Signal_to_Noise_Rate_3D
import lib.utils.metrics as Metrics
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator
import numpy as np
import copy
import torch
import time
import torch.optim as optim
import tqdm

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
    parse.add_argument('--resultdir', type=str, default='', dest='resultdir',
                     help='dir to save result')
    #parse.add_argument('--pretrain', action=argparse.BooleanOptionalAction)
    parse.add_argument('--model_to_test',default="0", dest='model_to_test', type=str ,
                        help='select model to test, autoencoder or auto_up')
    args = parse.parse_args()
    return args

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
"""
def ct_unGaussian(opt, value):
    return value * opt.CT_MEAN_STD[1] + opt.CT_MEAN_STD[0]

def ct_Gaussian(opt, value):
    return (value - opt.CT_MEAN_STD[0]) / opt.CT_MEAN_STD[1]

def metrics_evaluation(gt,pred):
    # 3D metrics including mse, cs and psnr
    #g_fake_unNorm = ct_unGaussian(pred)
    #g_real_unNorm = ct_unGaussian(gt)
    metrics_Mse = Metrics.Mean_Squared_Error(pred, gt)
    metrics_CosineSimilarity = Metrics.Cosine_Similarity(pred, gt)
    metrics_PSNR = Metrics.Peak_Signal_to_Noise_Rate(pred, gt, PIXEL_MAX=1.0)
    return metrics_Mse, metrics_CosineSimilarity, metrics_PSNR





def generateEvalMetrics(gt,pred,path):
        #Insert my pred
        #
        # Evaluate Part
        #
        generate_CT = visuals['G_fake'].data.clone().cpu().numpy()
        real_CT = visuals['G_real'].data.clone().cpu().numpy()
        # To [0, 1]
        # To NDHW
        if 'std' in opt.dataset_class or 'baseline' in opt.dataset_class:
            generate_CT_transpose = generate_CT
            real_CT_transpose = real_CT
        else:
            generate_CT_transpose = np.transpose(generate_CT, (0, 2, 1, 3))
            real_CT_transpose = np.transpose(real_CT, (0, 2, 1, 3))
        generate_CT_transpose = tensor_back_to_unnormalization(generate_CT_transpose, opt.CT_MEAN_STD[0],
                                                            opt.CT_MEAN_STD[1])
        real_CT_transpose = tensor_back_to_unnormalization(real_CT_transpose, opt.CT_MEAN_STD[0], opt.CT_MEAN_STD[1])
        # clip generate_CT
        generate_CT_transpose = np.clip(generate_CT_transpose, 0, 1)

        # CT range 0-1
        mae0 = MAE(real_CT_transpose, generate_CT_transpose, size_average=False)
        mse0 = MSE(real_CT_transpose, generate_CT_transpose, size_average=False)
        cosinesimilarity = Cosine_Similarity(real_CT_transpose, generate_CT_transpose, size_average=False)
        ssim = Structural_Similarity(real_CT_transpose, generate_CT_transpose, size_average=False, PIXEL_MAX=1.0)
        # CT range 0-4096
        generate_CT_transpose = tensor_back_to_unMinMax(generate_CT_transpose, opt.CT_MIN_MAX[0], opt.CT_MIN_MAX[1]).astype(np.int32)
        real_CT_transpose = tensor_back_to_unMinMax(real_CT_transpose, opt.CT_MIN_MAX[0], opt.CT_MIN_MAX[1]).astype(np.int32)
        psnr_3d = Peak_Signal_to_Noise_Rate_3D(real_CT_transpose, generate_CT_transpose, size_average=False, PIXEL_MAX=4095)
        psnr = Peak_Signal_to_Noise_Rate(real_CT_transpose, generate_CT_transpose, size_average=False, PIXEL_MAX=4095)
        mae = MAE(real_CT_transpose, generate_CT_transpose, size_average=False)
        mse = MSE(real_CT_transpose, generate_CT_transpose, size_average=False)

        name1 = os.path.splitext(os.path.basename(img_path[0][0]))[0]
        name2 = os.path.split(os.path.dirname(img_path[0][0]))[-1]
        name = name2 + '_' + name1
        print(cosinesimilarity, name)
        if cosinesimilarity is np.nan or cosinesimilarity > 1:
            print(os.path.splitext(os.path.basename(gan_model.get_image_paths()[0][0]))[0])
            continue

        metrics_list = [('MAE0', mae0), ('MSE0', mse0), ('MAE', mae), ('MSE', mse), ('CosineSimilarity', cosinesimilarity),
                        ('psnr-3d', psnr_3d), ('PSNR-1', psnr[0]),
                        ('PSNR-2', psnr[1]), ('PSNR-3', psnr[2]), ('PSNR-avg', psnr[3]),
                        ('SSIM-1', ssim[0]), ('SSIM-2', ssim[1]), ('SSIM-3', ssim[2]), ('SSIM-avg', ssim[3])]

        for key, value in metrics_list:
            if avg_dict.get(key) is None:
                avg_dict[key] = [] + value.tolist()
            else:
                avg_dict[key].extend(value.tolist())

        del visuals, img_path

    for key, value in avg_dict.items():
        print('### --{}-- total: {}; avg: {} '.format(key, len(value), np.round(np.mean(value), 7)))
        avg_dict[key] = np.mean(value)

"""

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

    
    print("Now testing: {} ".format(opt.model_to_test))
    

    print(opt.dataset_class)
    
    datasetClass, _, dataTestClass, collateClass = get_dataset(opt.dataset_class)
    opt.data_augmentation = dataTestClass
    dataset = datasetClass(opt)
    #dataset = dataTestClass(opt)


    if opt.model_to_test == "autoencoder":

    

        #feature_map_path = os.path.join(opt.MODEL_SAVE_PATH,"feature_map")
        #print(feature_map_path)
        


        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        autoencoder = ResUNet(in_channel=1,out_channel=1,training=False).to(device)
        autoencoder_path = os.path.join(opt.MODEL_SAVE_PATH,"autoencoder.pt")
        autoencoder.load_state_dict(torch.load(autoencoder_path))
        autoencoder.eval()

        #auto_down = ResUNet_Down(in_channel = 1, out_channel=256).to(device)
        #auto_down_path = os.path.join(opt.MODEL_SAVE_PATH,"auto_down.pt")
        #auto_down.load_state_dict(auto_down_path)
        
        print(autoencoder)
        
        #print(auto_down)
        # set to train

        pretrain_auto = {}
    
        #first batch size was 30
        #pretrain_auto["batch-print"] = 35
        
        pretrain_auto["batch_size"] = 1  #max batch is 13
        pretrain_auto["loss"] = torch.nn.L1Loss().to(device)
        #Gamma for next three items are 0.9
        #0.0001 err 0.0744 epoch 0 batch 30
        #0.00001 err 0.06 ep 0 batch 30 0.05-0.07
        #lr = 0.0005 batch 35 err 0.043-0.05

        #  for batch size 1, lr=0.00005
        
        #pretrain_auto["scheduler"] = optim.lr_scheduler.LamdaLR(pretrain_auto["optimizer"],batch_learn)
        

        

        dataloader_auto = torch.utils.data.DataLoader(
            dataset,
            batch_size= pretrain_auto['batch_size'],
            shuffle=False,
            num_workers=int(opt.nThreads),
            collate_fn=collateClass)
        avg_losses = []
        lr_list = []
        metrics_dict = {}
        predicts = None
    
        template = None
        pretrain_auto["running_loss"] = 0.0
        
    
        
        pretrain_auto["running_loss"] = 0.0
        pretrain_auto["running_ssim_avg"] = 0.0
        pretrain_auto["running_psnr"] = 0.0
        pretrain_auto["running_cos"] = 0.0
        
        for i, data in tqdm.tqdm(enumerate(dataloader_auto)):
            with torch.no_grad():

                X = data[0]
                if template == None:
                    template = data[3][0]
                    
                X = torch.unsqueeze(X,1)
                    
                X = X.to(device)
                
                pred = autoencoder(X)

                loss3 = pretrain_auto["loss"](pred,X)
                #print("\n loss0: {}, loss1: {}, loss2: {}, loss3: {} \n".format(loss0.item(),loss1.item(),loss2.item(),loss3.item()))
                loss = loss3
                
                
                metrics_dict["ssim_metric"] = Structural_Similarity(torch.squeeze(pred,dim=1).cpu().detach().numpy(),torch.squeeze(X,dim=1).cpu().detach().numpy())
                metrics_dict["cos_metric"] = Cosine_Similarity(torch.squeeze(pred,dim=1).cpu().detach().numpy(),torch.squeeze(X,dim=1).cpu().detach().numpy())
                metrics_dict["psnr_metric"] = Peak_Signal_to_Noise_Rate_3D(torch.squeeze(pred,dim=1).cpu().detach().numpy(),torch.squeeze(X,dim=1).cpu().detach().numpy())

                #MAE, MSE, Peak_Signal_to_Noise_Rate, Structural_Similarity, Cosine_Similarity, \
                #Peak_Signal_to_Noise_Rate_3D
                
                
                pretrain_auto["running_loss"] = pretrain_auto["running_loss"] + loss.item()
                pretrain_auto["running_ssim_avg"] = pretrain_auto["running_ssim_avg"] +metrics_dict["ssim_metric"][3]
                pretrain_auto["running_psnr"] = pretrain_auto["running_psnr"]+ metrics_dict["psnr_metric"]
                pretrain_auto["running_cos"] = pretrain_auto["running_cos"]+metrics_dict["cos_metric"]
                
                print("\n Loss: {}, ssim-avg: {}, cosine similarity = {},psnr = {}, Batch {}\n ".format(loss.item(),metrics_dict["ssim_metric"][3], 
                    metrics_dict["cos_metric"], metrics_dict["psnr_metric"], i))
        tot_metrics = {}
        
        tot_l1_loss = pretrain_auto["running_loss"]/len(dataloader_auto)
        tot_ssim = pretrain_auto["running_ssim_avg"]/len(dataloader_auto)
        tot_psnr = pretrain_auto["running_psnr"]/len(dataloader_auto)
        tor_cos = pretrain_auto["running_cos"]/len(dataloader_auto)

        tot_metrics["tot_l1_loss"] = tot_l1_loss
        tot_metrics["tot_ssim"] = tot_ssim
        tot_metrics["tot_psnr"] = tot_psnr
        tot_metrics["tor_cos"] = tor_cos
        print("\n")
        print(tot_metrics)
       
        #autoencoder_figs_path = os.path.join(opt.MODEL_SAVE_PATH,"figs","autoencoder","test")
        #avg_loss_path = os.path.join(autoencoder_figs_path,"avg-loss.png")
        #lr_path = os.path.join(autoencoder_figs_path,"lr.png")
        
        #print(feature_map.size())
        #print(dim)

   

        exit()
    else:
        print_easy_dict(opt)

        exit()
        
        feature_map = torch.load(feature_map_path +"\\feature_map.pt")
        long_range1 = torch.load(feature_map_path +"\\long_range1.pt")
        long_range2 = torch.load(feature_map_path +"\\long_range2.pt")
        long_range3 = torch.load(feature_map_path +"\\long_range3.pt")
        long_range4 = torch.load(feature_map_path +"\\long_range4.pt")
        #w = torch.tensor([0.5,0.5],requires_grad=False).to(device)
        auto_up = ResUNet_up(in_channel=256,out_channel=1).to(device)
       
            
        print(auto_up)
        
            #Load pretrained X2CT model
            #define and learn weights between X2CT and ResNet outputs
        gan_model = get_model(opt.model_class)()
        print('Model --{}-- will be Used'.format(gan_model.name))

        

        # set to test
        gan_model.eval()

        gan_model.init_process(opt)
                #total_steps, epoch_count = gan_model.setup(opt)
                
                #New training loop for both X2CT and auto_up

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

        result_dir = os.path.join(opt.resultdir, opt.data, '%s_%s' % (opt.dataset, opt.check_point))
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)

        avg_dict = dict()

        epochs = 1
        
        
        loss = torch.nn.L1Loss().to(device)
        optimizer = optim.Adam(auto_up.parameters(),lr=0.00005)

        scheduler = optim.lr_scheduler.ExponentialLR(optimizer,gamma=0.4, verbose=True)
        

        dataloader_decoder = torch.utils.data.DataLoader(
        dataset,
        batch_size= 10,
        shuffle=True,
        num_workers=int(opt.nThreads),
        collate_fn=collateClass)

        
        

        for epoch in range(epochs):
            for i,data in enumerate(dataloader_decoder):
                
            #maybe turn of grad on data
                if epoch == 1:
                    print(data)
                    print(data.size())
                
                gan_model.set_input(data)
                gan_model.test()
                visuals = gan_model.get_current_visuals()
                img_path = gan_model.get_image_paths()
                pred_G = gan_model.get_prediction()
                pred_res = auto_up(feature_map,pred_G=pred_G,
                    long_range1= long_range1,long_range2=long_range2,long_range3=long_range3,long_range4=long_range4)
                gt= gan_model.get_real()
                pred_res = torch.squeeze(pred_res,dim=0)
                optimizer.zero_grad()
                #wreap auto_up and FC layer for weight calibration
                #in the same class and add  self.weights = torch.nn.parameter.Parameter(weights) to it
                #Or just implement everything in auto_up class
                print("gt: {}".format(gt.size()))
                print("pred {}".format(pred_res.size()))
                cost = loss(gt,pred_res)
                cost.backward()
                optimizer.step()
                #auto_up.w = auto_up.w.detach()
                print("W: {}".format(auto_up.w))
                print("error: {}\n".format(cost.item()))
            scheduler.step()
           





        
        #print(pred)
        #print(torch.sum(pred))











    




    
    #print(feature_map)
    #print("SUM of feature map: {}".format(torch.sum(feature_map)))
   
