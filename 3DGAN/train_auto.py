
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
    parse.add_argument('--pretrain',default="0", dest='pretrain', type=str ,
                        help='if specified, pretrains autoencoder. If not trains pretrained models')
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
    if opt.pretrain == True:
        print_easy_dict(opt)

    opt.pretrain = str2bool(opt.pretrain)
    print("Pretraining: {} ".format(opt.pretrain))
    

    
    datasetClass, augmentationClass, dataTestClass, collateClass = get_dataset(opt.dataset_class)
    opt.data_augmentation = augmentationClass
    dataset = datasetClass(opt)


    
    

    feature_map_path = os.path.join(opt.MODEL_SAVE_PATH,"feature_map")
    #print(feature_map_path)
    


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
    if opt.pretrain == True:
        print(autoencoder)

    



    dataloader_auto = torch.utils.data.DataLoader(
        dataset,
        batch_size= 1,
        shuffle=True,
        num_workers=int(opt.nThreads),
        collate_fn=collateClass)

    predicts = None
    #remove below line when not debugging
    pretrain_auto["epoch"] = 2
    template = None

    #if pretrian != None
    if opt.pretrain == True:
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
        print(auto_down)
        feature_map = auto_down(template)
        dim = feature_map.size()[2] 
        #print(feature_map.size())
        #print(dim)
        if os.path.isfile(os.path.join(feature_map_path,"feature_map.pt")):
            os.remove(os.path.join(feature_map_path,"feature_map.pt"))
        
        file_path = os.path.join(feature_map_path,"feature_map.pt")
        
        torch.save(feature_map,file_path)
        exit()
    else:
        


        
        feature_map = torch.load(feature_map_path +"\\feature_map.pt")
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
                pred_res = auto_up(feature_map,pred_G=pred_G)
                gt= gan_model.get_real()
                optimizer.zero_grad()
                #wreap auto_up and FC layer for weight calibration
                #in the same class and add  self.weights = torch.nn.parameter.Parameter(weights) to it
                #Or just implement everything in auto_up class
                cost = loss(gt,pred_res)
                cost.backward()
                optimizer.step()
                #auto_up.w = auto_up.w.detach()
                print("W", auto_up.w.shape)
                print(auto_up.w)
                print("error: {}\n".format(cost.item()))
            scheduler.step()




        
        #print(pred)
        #print(torch.sum(pred))



        """
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







    




    
    #print(feature_map)
    #print("SUM of feature map: {}".format(torch.sum(feature_map)))
   
