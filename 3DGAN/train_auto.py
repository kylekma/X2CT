
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
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator
import copy
import torch
import time
import torch.optim as optim


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
    #parse.add_argument('--model_to_train',default="0", dest='model_to_train', type=str ,
     #                   help='select model to train, decodeCorrection or decodeGroundTruth')
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
    
    print_easy_dict(opt)
    

    opt.pretrain = str2bool(opt.pretrain)
    print("Pretraining: {} ".format(opt.pretrain))
    #if opt.model_to_train == "0" and opt.pretrain == False:
    #    print("Please select model to train: decodeCorrection or decodeGroundTruth")
    #    exit(0)
    #opt.model_to_train = opt.model_to_train.lower()

    

    
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
    pretrain_auto["lr"] = lr=0.00075
    pretrain_auto["alpha"] = 0.4
    pretrain_auto["epoch"] = 25
    #first batch size was 30
    pretrain_auto["batch-print"] = 35
    pretrain_auto["batch_size"] = 5
    pretrain_auto["loss"] = torch.nn.L1Loss().to(device)
    #Gamma for next three items are 0.9
    #0.0001 err 0.0744 epoch 0 batch 30
    #0.00001 err 0.06 ep 0 batch 30 0.05-0.07
    #lr = 0.0005 batch 35 err 0.043-0.05

    #  for batch size 1, lr=0.00005
    pretrain_auto["optimizer"] = optim.Adam(autoencoder.parameters(),lr=0.00075)

    pretrain_auto["scheduler"] = optim.lr_scheduler.ExponentialLR(pretrain_auto["optimizer"],gamma=0.6, verbose=True)
    #pretrain_auto["scheduler"] = optim.lr_scheduler.LamdaLR(pretrain_auto["optimizer"],batch_learn)
    if opt.pretrain == True:
        print(autoencoder)

    print("figs path: {}".format(os.path.join(opt.MODEL_SAVE_PATH,"figs","autoencoder","train")))

    dataloader_auto = torch.utils.data.DataLoader(
        dataset,
        batch_size= pretrain_auto['batch_size'],
        shuffle=True,
        num_workers=int(opt.nThreads),
        collate_fn=collateClass)
    avg_losses = []
    lr_list = []
    predicts = None
   
    template = None
    pretrain_auto["running_loss"] = 0.0
    curr_lr = 0
    #if pretrian != None
    if opt.pretrain == True:
    #pretraining of autoencoder
        for epoch in range(pretrain_auto["epoch"]):
            pretrain_auto["running_loss"] = 0.0
            correct = 0
            for i, data in enumerate(dataloader_auto):
                X = data[0]
                if template == None:
                    template = data[3][0]
                    
                X = torch.unsqueeze(X,1)
                    
                X = X.to(device)

                pretrain_auto["optimizer"].zero_grad()
                
                predicts = autoencoder(X)

                loss0 = pretrain_auto["loss"](predicts[0],X)
                loss1 = pretrain_auto["loss"](predicts[1],X)
                loss2 = pretrain_auto["loss"](predicts[2],X)
                loss3 = pretrain_auto["loss"](predicts[3],X)
                #print("\n loss0: {}, loss1: {}, loss2: {}, loss3: {} \n".format(loss0.item(),loss1.item(),loss2.item(),loss3.item()))
                loss = loss3 + (pretrain_auto["alpha"] *(loss0 + loss1 + loss2))
                loss.backward()
                pretrain_auto["optimizer"].step()
                pretrain_auto["running_loss"] = pretrain_auto["running_loss"] + loss.item()
                if i % pretrain_auto["batch-print"] == 0:
                    print("\n Epoch: {}, Loss: {}, Batch {}\n ".format(epoch, loss.item(),i))
            curr_lr = pretrain_auto["scheduler"].optimizer.param_groups[0]['lr']
            pretrain_auto["scheduler"].step()
            avg_loss = pretrain_auto["running_loss"]/len(dataloader_auto)
            avg_losses.append(avg_loss)
            lr_list.append(curr_lr)

            


        

        
        
        template = template.to(device)

        template = torch.unsqueeze(template,dim=0)
        template = torch.unsqueeze(template,dim=0)
        keys = set(auto_down.state_dict().keys())
        auto_down.load_state_dict({k:v for k,v in autoencoder.state_dict().items() if k in keys})
        print(auto_down)
        feature_map,long_range1, long_range2, long_range3, long_range4 = auto_down(template)
        dim = feature_map.size()[2] 
        #print(feature_map.size())
        #print(dim)
        file_path = os.path.join(feature_map_path,"feature_map.pt")
        long_range1_path = os.path.join(feature_map_path,"long_range1.pt")
        long_range2_path = os.path.join(feature_map_path,"long_range2.pt")
        long_range3_path = os.path.join(feature_map_path,"long_range3.pt")
        long_range4_path = os.path.join(feature_map_path,"long_range4.pt")
        autoencoder_figs_path = os.path.join(opt.MODEL_SAVE_PATH,"figs","autoencoder","train")
        avg_loss_path = os.path.join(autoencoder_figs_path,"avg-loss.png")
        lr_path = os.path.join(autoencoder_figs_path,"lr.png")
        
        
        autoencoder_path = os.path.join(opt.MODEL_SAVE_PATH,"autoencoder.pt")
        auto_down_path = os.path.join(opt.MODEL_SAVE_PATH,"auto_down.pt")

        if os.path.isfile(file_path):
            os.remove(file_path)
        if os.path.isfile(long_range1_path):
            os.remove(long_range1_path)
        if os.path.isfile(long_range2_path):
            os.remove(long_range2_path)
        if os.path.isfile(long_range3_path):
            os.remove(long_range3_path)
        if os.path.isfile(long_range4_path):
            os.remove(long_range4_path)
        if os.path.isfile(avg_loss_path):
            os.remove(avg_loss_path)
        if os.path.isfile(lr_path):
            os.remove(lr_path)
        if os.path.isfile(autoencoder_path):
            os.remove(autoencoder_path)
        if os.path.isfile(auto_down_path):
            os.remove(auto_down_path)
        

        

        

        torch.save(feature_map,file_path)
        torch.save(long_range1,long_range1_path)
        torch.save(long_range2,long_range2_path)
        torch.save(long_range3,long_range3_path)
        torch.save(long_range4,long_range4_path)
        torch.save(autoencoder.state_dict(),autoencoder_path)
        torch.save(auto_down.state_dict(),auto_down_path)

        fg_loss = Figure()
        ax_loss = fg_loss.gca()
        ax_loss.plot(avg_losses)
        ax_loss.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax_loss.set_xlabel('epochs', fontsize=10)
        ax_loss.set_ylabel('avg-loss', fontsize='medium')
        
        fg_loss.savefig(avg_loss_path)

        fg_lr = Figure()
        ax_lr = fg_lr.gca()
        ax_lr.plot(lr_list)
        ax_lr.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax_lr.set_xlabel('epochs', fontsize=10)
        ax_lr.set_ylabel('lr', fontsize='medium')
        
        fg_lr.savefig(lr_path)

        exit()
    else:
        
        #create Template

        
        auto_down_path = os.path.join(opt.MODEL_SAVE_PATH,"auto_down.pt")
        auto_down = ResUNet_Down(in_channel = 1, out_channel=256).to(device)
        auto_down_weights = torch.load(auto_down_path)
        auto_down.load_state_dict(auto_down_weights)

        #feature_map = torch.load(feature_map_path +"\\feature_map.pt")
        #long_range1 = torch.load(feature_map_path +"\\long_range1.pt")
        #long_range2 = torch.load(feature_map_path +"\\long_range2.pt")
        #long_range3 = torch.load(feature_map_path +"\\long_range3.pt")
        #long_range4 = torch.load(feature_map_path +"\\long_range4.pt")
        #w = torch.tensor([0.5,0.5],requires_grad=False).to(device)
        auto_up = ResUNet_up(in_channel=1,out_channel=1, training=True, pretrained = auto_down).to(device)
       
            
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
        batch_size = 5
        lr = 0.00015
        gamma = 0.4
        running_loss = 0.0
        curr_lr = 0.0
        avg_losses = []
        lr_list = []
        encoder_fmap = torch.empty()
        decoder_fmap = torch.empty()
        
        loss = torch.nn.L1Loss().to(device)
        optimizer = optim.Adam(auto_up.parameters(),lr)

        scheduler = optim.lr_scheduler.ExponentialLR(optimizer,gamma, verbose=True)
        
        dataloader_decoder = torch.utils.data.DataLoader(
        dataset,
        batch_size= batch_size,
        shuffle=True,
        num_workers=int(opt.nThreads),
        collate_fn=collateClass)


        #Train decodeCorrection
    

        for epoch in range(epochs):
            running_loss = 0.0
            for i,data in enumerate(dataloader_decoder):
                
                gan_model.set_input(data)
                gan_model.test()
                visuals = gan_model.get_current_visuals()
                img_path = gan_model.get_image_paths()
                pred_G = gan_model.get_prediction()
                #template =
                pred_res, encoder_fmap, decoder_fmap= auto_up(template)
                gt= gan_model.get_real()
                pred_res = torch.squeeze(pred_res,dim=0)
                optimizer.zero_grad()
                print("gt: {}".format(gt.size()))
                print("pred {}".format(pred_res.size()))
                cost = loss(gt,pred_res)
                cost.backward()
                optimizer.step()
                #print("W: {}".format(auto_up.w))
                print("\n Epoch: {}, Loss: {}, Batch {}\n ".format(epoch, cost.item(),i))
            curr_lr = scheduler.optimizer.param_groups[0]['lr']
            scheduler.step()
            avg_loss = running_loss/len(dataloader_auto)
            avg_losses.append(avg_loss)
            lr_list.append(curr_lr)

        feature_map_path = os.path.join(opt.MODEL_SAVE_PATH,"feature_map")
        auto_up_path = os.path.join(opt.MODEL_SAVE_PATH,"auto_down.pt")
        encoder_fmap_path = os.path.join(feature_map_path,"encoder_fmap.pt")
        decoder_fmap_path = os.path.join(feature_map_path,"decoder_fmap.pt")


        if os.path.isfile(auto_up_path):
            os.remove(auto_up_path)
        if os.path.isfile(encoder_fmap_path):
            os.remove(encoder_fmap_path)
        if os.path.isfile(decoder_fmap_path):
            os.remove(decoder_fmap_path)

        
        torch.save(auto_up.state_dict(),auto_up_path)
        torch.save(encoder_fmap,encoder_fmap_path)
        torch.save(decoder_fmap,decoder_fmap_path)

        #save encoder_fmap and decoder_fmap on disk


            

           







      







    




    
    #print(feature_map)
    #print("SUM of feature map: {}".format(torch.sum(feature_map)))
   
