
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
    parse.add_argument('--pretrain', action='store_true',default=None, dest='pretrain',
                        help='if specified, pretrains autoencoder. If not trains pretrained models')
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

    
    datasetClass, augmentationClass, dataTestClass, collateClass = get_dataset(opt.dataset_class)
    opt.data_augmentation = augmentationClass
    dataset = datasetClass(opt)







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
    print(autoencoder)

    



    dataloader_auto = torch.utils.data.DataLoader(
        dataset,
        batch_size= 1,
        shuffle=True,
        num_workers=int(opt.nThreads),
        collate_fn=collateClass)

    predicts = None
    #remove below line when not debugging
    pretrain_auto["epoch"] = 1
    template = None

    #if pretrian != None
    if opt.pretrain:
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
            
            auto_up = ResUNet_up(in_channel=256,out_channel=1,dim= dim).to(device)
            
            print(auto_up)
            output = auto_up(feature_map)
            #Load pretrained X2CT model
            #define and learn weights between X2CT and ResNet outputs
            gan_model = get_model(opt.model_class)()
            print('Model --{}-- will be Used'.format(gan_model.name))

            # set to test
            gan_model.eval()

            gan_model.init_process(opt)
            #total_steps, epoch_count = gan_model.setup(opt)
            
            #New training loop for both X2CT and auto_up


    




    
    #print(feature_map)
    #print("SUM of feature map: {}".format(torch.sum(feature_map)))
   
