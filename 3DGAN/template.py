import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import torch
from lib.dataset.factory import get_dataset
import SimpleITK as sitk
import argparse
import h5py
import numpy as np

class optDict():
    def __init__(self, dataset_class):
        self.dataset_class = dataset_class
        self.datasetfile = "./data/train.txt"
        self.dataroot = "./data/LIDC-HDF5-256"
        self.ct_channel = 128
        self.fine_size = 128
        self.resize_size= 150
        self.xray_channel= 1
        self.CT_MIN_MAX= [0, 2500]
        self.XRAY1_MIN_MAX= [0, 255]
        self.XRAY2_MIN_MAX= [0, 255]
        self.CT_MEAN_STD= [0.0, 1.0]
        self.XRAY1_MEAN_STD= [0.0, 1.0]
        self.XRAY2_MEAN_STD= [0.0, 1.0]
        self.nThreads= 5                                                       #3DGAN
        self.TEMPLATE_DATA_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__),"data", "deepreg-data"))

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

    parser = argparse.ArgumentParser()
    parser = argparse.ArgumentParser()
    parser.add_argument(
    "--createData",
    help="Choose if you want to create data for template registration in data/deepreg-data/train/",
    dest="createData",
    action="store_true",
)
    args = parser.parse_args()
    args.createData = str2bool(args.createData)
    opt = optDict("align_ct_xray_views_std")
    train_path = os.path.join(opt.TEMPLATE_DATA_PATH,"train","images.h5")
    
    print()


    if args.createData:


        # merge config with yaml
        # merge config with argparse
        

        train_path = os.path.join(opt.TEMPLATE_DATA_PATH,"train","images.h5")
        print(train_path)
        N = 915 # find the length of my dataset
        h5_file = h5py.File(train_path,"w")

        h5_train = h5_file.create_dataset('data', shape=(N, 128, 128,128), dtype=np.float32, fillvalue=0)
        
        print(opt.dataset_class)

        
        datasetClass, augmentationClass, dataTestClass, collateClass = get_dataset(opt.dataset_class)
        opt.data_augmentation = augmentationClass


        dataset = datasetClass(opt)
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=1 ,#915 total
            shuffle=True,
            num_workers=int(opt.nThreads),
            collate_fn=collateClass)

        
        #dtype torch.float32
        for idx, data in enumerate(dataloader):
            ct = data[0]
            ct = torch.squeeze(ct,dim=0)
            h5_train[idx] = ct.numpy()
            
           # f = h5py.File("file{}.h5".format(idx), 'w')
        h5_file.close()
        



    h5_file = h5py.File(train_path,"r")
    print(h5_file.keys())
    h5_cont = h5_file.get("data")
    print(h5_cont.shape)
    for idx in range(len(h5_cont)):
        print(h5_cont[idx])
        break


    print(h5_cont[len(h5_cont)-1].shape)
    print(np.sum(h5_cont[len(h5_cont)-1]))
    h5_file.close()