"""
This code is referenced from https://github.com/assassint2017/MICCAI-LITS2017
"""
from __future__ import print_function
from __future__ import absolute_import
from __future__ import division
from turtle import forward
from numpy import short
import torch
import torch.nn as nn
import torch.nn.functional as F
from lib.model.base_model import Base_Model
import lib.model.nets.factory as factory





class ResUNet_up(nn.Module):
        #output1 = map1, output2 = map2, output3 = map3, output4 = map4
        #long_range1=16, long_range2=32, long_range3=64, long_range4=128
  
    def __init__(self,in_channel,out_channel,training, encoder):
        super(ResUNet_up,self).__init__()
        #self.w = torch.nn.Parameter(torch.tensor([0.5,0.5]))
        self.drop_rate = 0.2
        self.training = training

        self.encoder = encoder

        #All sizes of channels and width height and depht are divided by 2
        #Even longranges are maxpooled and down_conv so to halv (c,w,h,d)
        self.down_conv = nn.Sequential(
            nn.Conv3d(256,int(256/2),kernel_size=3,stride=1,padding=1),
            nn.PReLU(int(256/2)),
            nn.MaxPool3d(2,stride=2, padding=0),
            nn.BatchNorm3d(int(256/2))
        )

 
        
        self.up_conv2 = nn.Sequential(
            nn.ConvTranspose3d(128, 64, 2, 2),
            nn.PReLU(64)
        )

        self.up_conv3 = nn.Sequential(
            nn.ConvTranspose3d(64, 32, 2, 2),
            nn.PReLU(32)
        )

        self.up_conv4 = nn.Sequential(
            nn.ConvTranspose3d(32, 16, 2, 2),
            nn.PReLU(16)
        )
        self.conv_long_range1 = nn.Sequential(
            nn.Conv3d(16,8,kernel_size=3,stride=1,padding=1),
            nn.PReLU(8),
            nn.MaxPool3d(2,stride=2, padding=0),
            nn.BatchNorm3d(8)
            
        )
        self.conv_long_range2 = nn.Sequential(
            nn.Conv3d(32,16,kernel_size=3,stride=1,padding=1),
            nn.PReLU(16),

            nn.MaxPool3d(2,stride=2, padding=0)
            
        )
        self.conv_long_range3 = nn.Sequential(
            nn.Conv3d(64,32,kernel_size=3,stride=1,padding=1),
            nn.PReLU(32),
            nn.MaxPool3d(2,stride=2, padding=0),
            nn.BatchNorm3d(32)
        )
        self.conv_long_range4 = nn.Sequential(
            nn.Conv3d(128,64,kernel_size=3,stride=1,padding=1),
            nn.PReLU(64),
            nn.MaxPool3d(2,stride=2, padding=0),
            nn.BatchNorm3d(64)
        )
        self.decoder_stage1 = nn.Sequential(
            nn.Conv3d(64, 128, 3, 1, padding=1),
            nn.PReLU(128),

            nn.Conv3d(128, 128, 3, 1, padding=1),
            nn.PReLU(128),

            nn.Conv3d(128, 128, 3, 1, padding=1),
            nn.PReLU(128),
        )

        self.decoder_stage2 = nn.Sequential(
            nn.Conv3d(64+32, 64, 3, 1, padding=1),
            nn.PReLU(64),

            nn.Conv3d(64, 64, 3, 1, padding=1),
            nn.PReLU(64),

            nn.Conv3d(64, 64, 3, 1, padding=1),
            nn.PReLU(64),
        )

        self.decoder_stage3 = nn.Sequential(
            nn.Conv3d(32 +16, 32, 3, 1, padding=1),
            nn.PReLU(32),

            nn.Conv3d(32, 32, 3, 1, padding=1),
            nn.PReLU(32),

            nn.Conv3d(32, 32, 3, 1, padding=1),
            nn.PReLU(32),
        )
        self.decoder_stage4 = nn.Sequential(
            nn.Conv3d(16+8, 16, 3, 1, padding=1),
            nn.PReLU(16),

            nn.Conv3d(16, 16, 3, 1, padding=1),
            nn.PReLU(16),
        )
        
        self.long_range4_map = nn.Sequential(
            nn.Conv3d(128, out_channel, 1, 1),
            nn.Upsample(scale_factor=(2, 2, 2), mode='trilinear', align_corners=False),
        )
        
        self.long_range3_map = nn.Sequential(
            nn.Conv3d(64, out_channel, 1, 1),
            nn.Upsample(scale_factor=(4, 4, 4), mode='trilinear', align_corners=False),
        )

        
        self.long_range2_map = nn.Sequential(
            nn.Conv3d(32, out_channel, 1, 1),
            nn.Upsample(scale_factor=(8, 8, 8), mode='trilinear', align_corners=False),
        )

        
        self.long_range1_map = nn.Sequential(
            nn.Conv3d(16, out_channel, 1, 1),
            nn.Upsample(scale_factor=(16, 16, 16), mode='trilinear', align_corners=False),
           
        )

        
        self.map4 = nn.Sequential(
            nn.Conv3d(16, out_channel, 1, 1),
            nn.Upsample(scale_factor=(2, 2, 2), mode='trilinear', align_corners=False),
        )
        
        self.map3 = nn.Sequential(
            nn.Conv3d(32, out_channel, 1, 1),
            nn.Upsample(scale_factor=(4, 4, 4), mode='trilinear', align_corners=False),
        )

        
        self.map2 = nn.Sequential(
            nn.Conv3d(64, out_channel, 1, 1),
            nn.Upsample(scale_factor=(8, 8, 8), mode='trilinear', align_corners=False),
        )

        
        self.map1 = nn.Sequential(
            nn.Conv3d(128, out_channel, 1, 1),
            nn.Upsample(scale_factor=(16, 16, 16), mode='trilinear', align_corners=False),
           
        )
        
    def forward(self,X):
        #long_range1=16, long_range2=32, long_range3=64, long_range4=128
        #print("longrange4: {}".format(long_range4.size()))
       #print("before")
        #print("long_rage1: {},long_rage2: {},long_rage3: {},long_rage4: {}".format(long_range1.size(),long_range2.size(),long_range3.size(),long_range4.size()))
        short_range4, long_range1_notc, long_range2_notc,long_range3_notc,long_range4_notc = self.encoder(X)
        long_range1 = self.conv_long_range1(long_range1_notc)
        long_range2 = self.conv_long_range2(long_range2_notc)
        long_range3 = self.conv_long_range3(long_range3_notc)
        long_range4 = self.conv_long_range4(long_range4_notc)
        
        #print("long_rage1: {},long_rage2: {},long_rage3: {},long_rage4: {}".format(long_range1.size(),long_range2.size(),long_range3.size(),long_range4.size()))
        
        short_range4 = self.down_conv(short_range4) 

    
        outputs = self.decoder_stage1(long_range4) + short_range4 
        decoder_feature1 = outputs

        outputs = F.dropout(outputs, self.drop_rate, self.training)
        
        
        output1 = self.map1(outputs) 
       
        short_range6 = self.up_conv2(outputs)
     
 
        outputs = self.decoder_stage2(torch.cat([short_range6, long_range3], dim=1)) + short_range6 
        decoder_feature2 = outputs

        outputs = F.dropout(outputs, self.drop_rate, self.training)
       
     
        
        output2 = self.map2(outputs) 
        
        
        short_range7 = self.up_conv3(outputs)

        outputs = self.decoder_stage3(torch.cat([short_range7, long_range2], dim=1)) + short_range7 #add batchnormlayer after

        decoder_feature3 = outputs

        outputs = F.dropout(outputs, self.drop_rate, self.training)

        output3 = self.map3(outputs)

        short_range8 = self.up_conv4(outputs)

        outputs = self.decoder_stage4(torch.cat([short_range8, long_range1], dim=1)) + short_range8 #add batchnormlayer after

        decoder_feature4 = outputs
        
        output4 = self.map4(outputs)
        
        
        """
        short_range6 = self.up_conv2(X)
        #print(X.size())
        X = self.decoder_stage2(torch.cat([short_range6,long_range4],dim=1)) + short_range6
        X = F.dropout(X, self.drop_rate, self.training)
        #print(X.size())

        short_range7 = self.up_conv3(X)
        #print(X.size())

        X = self.decoder_stage3(short_range7) + short_range7
        X = F.dropout(X, self.drop_rate, self.training)
        #print(X.size())

        short_range8 = self.up_conv4(X)
        #print(X.size())

        X = self.decoder_stage4(short_range8) + short_range8
        X = self.map4(X)
        #print(self.w.size())

        """
        
        #w = self.w
        #print("calibr weights: {}".format(self.w))
        if self.training is True:
            output4 = output1 + output2 + output3 + output4
            #output4 = output4*w[0] + pred_G * w[1]
            output4 = output4 + pred_G
        else:
            #output4 = output4*w[0] + pred_G * w[1]
            output4 = output4 + pred_G
            encoder_fmaps = torch.tensor([long_range1_notc.clone().detach(),long_range2_notc.clone().detach(),long_range3_notc.clone().detach(),long_range4_notc.clone().detach()])
            decoder_fmaps = torch.tensor([decoder_feature1,decoder_feature2,decoder_feature3,decoder_feature4])



        return output4,encoder_fmaps,decoder_fmaps
        




class ResUNet(nn.Module):
    def __init__(self, in_channel=1, out_channel=1 ,training=True):
        super(ResUNet, self).__init__()

        self.training = training
        self.drop_rate = 0.2

        self.encoder_stage1 = nn.Sequential(
            nn.Conv3d(in_channel, 16, 3, 1, padding=1),
            nn.PReLU(16),

            nn.Conv3d(16, 16, 3, 1, padding=1),
            nn.PReLU(16),
        )

        self.encoder_stage2 = nn.Sequential(
            nn.Conv3d(32, 32, 3, 1, padding=1),
            nn.PReLU(32),

            nn.Conv3d(32, 32, 3, 1, padding=1),
            nn.PReLU(32),

            nn.Conv3d(32, 32, 3, 1, padding=1),
            nn.PReLU(32),
        )

        self.encoder_stage3 = nn.Sequential(
            nn.Conv3d(64, 64, 3, 1, padding=1),
            nn.PReLU(64),

            nn.Conv3d(64, 64, 3, 1, padding=2, dilation=2),
            nn.PReLU(64),

            nn.Conv3d(64, 64, 3, 1, padding=4, dilation=4),
            nn.PReLU(64),
        )

        self.encoder_stage4 = nn.Sequential(
            nn.Conv3d(128, 128, 3, 1, padding=3, dilation=3),
            nn.PReLU(128),

            nn.Conv3d(128, 128, 3, 1, padding=4, dilation=4),
            nn.PReLU(128),

            nn.Conv3d(128, 128, 3, 1, padding=5, dilation=5),
            nn.PReLU(128),
        )

        self.decoder_stage1 = nn.Sequential(
            nn.Conv3d(128, 256, 3, 1, padding=1),
            nn.PReLU(256),

            nn.Conv3d(256, 256, 3, 1, padding=1),
            nn.PReLU(256),

            nn.Conv3d(256, 256, 3, 1, padding=1),
            nn.PReLU(256),
        )

        self.decoder_stage2 = nn.Sequential(
            nn.Conv3d(128 + 64, 128, 3, 1, padding=1),
            nn.PReLU(128),

            nn.Conv3d(128, 128, 3, 1, padding=1),
            nn.PReLU(128),

            nn.Conv3d(128, 128, 3, 1, padding=1),
            nn.PReLU(128),
        )

        self.decoder_stage3 = nn.Sequential(
            nn.Conv3d(64 + 32, 64, 3, 1, padding=1),
            nn.PReLU(64),

            nn.Conv3d(64, 64, 3, 1, padding=1),
            nn.PReLU(64),

            nn.Conv3d(64, 64, 3, 1, padding=1),
            nn.PReLU(64),
        )

        self.decoder_stage4 = nn.Sequential(
            nn.Conv3d(32 + 16, 32, 3, 1, padding=1),
            nn.PReLU(32),

            nn.Conv3d(32, 32, 3, 1, padding=1),
            nn.PReLU(32),
        )

        self.down_conv1 = nn.Sequential(
            nn.Conv3d(16, 32, 2, 2),
            nn.PReLU(32)
        )

        self.down_conv2 = nn.Sequential(
            nn.Conv3d(32, 64, 2, 2),
            nn.PReLU(64)
        )

        self.down_conv3 = nn.Sequential(
            nn.Conv3d(64, 128, 2, 2),
            nn.PReLU(128)
        )

        self.down_conv4 = nn.Sequential(
            nn.Conv3d(128, 256, 3, 1, padding=1),
            nn.PReLU(256)
        )

        self.up_conv2 = nn.Sequential(
            nn.ConvTranspose3d(256, 128, 2, 2),
            nn.PReLU(128)
        )

        self.up_conv3 = nn.Sequential(
            nn.ConvTranspose3d(128, 64, 2, 2),
            nn.PReLU(64)
        )

        self.up_conv4 = nn.Sequential(
            nn.ConvTranspose3d(64, 32, 2, 2),
            nn.PReLU(32)
        )

        # 最后大尺度下的映射（256*256），下面的尺度依次递减
        self.map4 = nn.Sequential(
            nn.Conv3d(32, out_channel, 1, 1),
            nn.Upsample(scale_factor=(1, 1, 1), mode='trilinear', align_corners=False),
            #nn.Softmax(dim=1)
        )

        # 128*128 尺度下的映射
        self.map3 = nn.Sequential(
            nn.Conv3d(64, out_channel, 1, 1),
            nn.Upsample(scale_factor=(2, 2, 2), mode='trilinear', align_corners=False),
            #nn.Softmax(dim=1)
        )

        # 64*64 尺度下的映射
        self.map2 = nn.Sequential(
            nn.Conv3d(128, out_channel, 1, 1),
            nn.Upsample(scale_factor=(4, 4, 4), mode='trilinear', align_corners=False),

            #nn.Softmax(dim=1)
        )

        # 32*32 尺度下的映射
        self.map1 = nn.Sequential(
            nn.Conv3d(256, out_channel, 1, 1),
            nn.Upsample(scale_factor=(8, 8, 8), mode='trilinear', align_corners=False),
            #nn.Softmax(dim=1)
        )

        self.batch_norm1 = nn.BatchNorm3d(16)
        self.batch_norm2 = nn.BatchNorm3d(32)
        self.batch_norm3 = nn.BatchNorm3d(64)
        self.batch_norm4 = nn.BatchNorm3d(128)
    @property
    def name(self):
        return 'multiView_AutoEncoder'

    def forward(self, inputs):

        #print("input")
        #print(str(inputs.size()))

        long_range1 = self.encoder_stage1(inputs) + inputs
        long_range1 = self.batch_norm1(long_range1)

        short_range1 = self.down_conv1(long_range1)

        long_range2 = self.encoder_stage2(short_range1) + short_range1
        long_range2 = self.batch_norm2(long_range2)
        long_range2 = F.dropout(long_range2, self.drop_rate, self.training)

        short_range2 = self.down_conv2(long_range2)

        long_range3 = self.encoder_stage3(short_range2) + short_range2
        long_range3 = self.batch_norm3(long_range3)

        long_range3 = F.dropout(long_range3, self.drop_rate, self.training)

        short_range3 = self.down_conv3(long_range3)

        long_range4 = self.encoder_stage4(short_range3) + short_range3
        long_range4 = self.batch_norm4(long_range4)
        long_range4 = F.dropout(long_range4, self.drop_rate, self.training)

        short_range4 = self.down_conv4(long_range4) #128 to 256

        outputs = self.decoder_stage1(long_range4) + short_range4 #128 to 265
        outputs = F.dropout(outputs, self.drop_rate, self.training)

        output1 = self.map1(outputs)

        short_range6 = self.up_conv2(outputs)

        outputs = self.decoder_stage2(torch.cat([short_range6, long_range3], dim=1)) + short_range6
        outputs = F.dropout(outputs, self.drop_rate, self.training)

        output2 = self.map2(outputs)

        short_range7 = self.up_conv3(outputs)

        outputs = self.decoder_stage3(torch.cat([short_range7, long_range2], dim=1)) + short_range7
        outputs = F.dropout(outputs, self.drop_rate, self.training)

        output3 = self.map3(outputs)

        short_range8 = self.up_conv4(outputs)

        outputs = self.decoder_stage4(torch.cat([short_range8, long_range1], dim=1)) + short_range8

        output4 = self.map4(outputs)

        #print("output1:")
        #print(str(output1.size()))
        #print("output2")
        #print(str(output2.size()))
        #print("output3")
        #print(str(output3.size()))
        #print("output4")
        #print(str(output4.size()))


        if self.training is True:
            return output1, output2, output3, output4
        else:
            return output4


class ResUNet_Down(nn.Module):
    def __init__(self, in_channel=1, out_channel=256 ,training=True):
        super().__init__()

        self.training = training
        self.drop_rate = 0.2

        self.encoder_stage1 = nn.Sequential(
            nn.Conv3d(in_channel, 16, 3, 1, padding=1),
            nn.PReLU(16),

            nn.Conv3d(16, 16, 3, 1, padding=1),
            nn.PReLU(16),
        )

        self.encoder_stage2 = nn.Sequential(
            nn.Conv3d(32, 32, 3, 1, padding=1),
            nn.PReLU(32),

            nn.Conv3d(32, 32, 3, 1, padding=1),
            nn.PReLU(32),

            nn.Conv3d(32, 32, 3, 1, padding=1),
            nn.PReLU(32),
        )

        self.encoder_stage3 = nn.Sequential(
            nn.Conv3d(64, 64, 3, 1, padding=1),
            nn.PReLU(64),

            nn.Conv3d(64, 64, 3, 1, padding=2, dilation=2),
            nn.PReLU(64),

            nn.Conv3d(64, 64, 3, 1, padding=4, dilation=4),
            nn.PReLU(64),
        )

        self.encoder_stage4 = nn.Sequential(
            nn.Conv3d(128, 128, 3, 1, padding=3, dilation=3),
            nn.PReLU(128),

            nn.Conv3d(128, 128, 3, 1, padding=4, dilation=4),
            nn.PReLU(128),

            nn.Conv3d(128, 128, 3, 1, padding=5, dilation=5),
            nn.PReLU(128),
        )

        self.down_conv1 = nn.Sequential(
            nn.Conv3d(16, 32, 2, 2),
            nn.PReLU(32)
        )

        self.down_conv2 = nn.Sequential(
            nn.Conv3d(32, 64, 2, 2),
            nn.PReLU(64)
        )

        self.down_conv3 = nn.Sequential(
            nn.Conv3d(64, 128, 2, 2),
            nn.PReLU(128)
        )

        self.down_conv4 = nn.Sequential(
            nn.Conv3d(128, out_channel, 3, 1, padding=1),
            nn.PReLU(256)
        )
        self.batch_norm1 = nn.BatchNorm3d(16)
        self.batch_norm2 = nn.BatchNorm3d(32)
        self.batch_norm3 = nn.BatchNorm3d(64)
        self.batch_norm4 = nn.BatchNorm3d(128)

        
    @property
    def name(self):
        return 'multiView_AutoEncoder'

    def forward(self, inputs):

        long_range1 = self.encoder_stage1(inputs) + inputs
        long_range1 = self.batch_norm1(long_range1)

        short_range1 = self.down_conv1(long_range1)

        long_range2 = self.encoder_stage2(short_range1) + short_range1
        long_range2 = self.batch_norm2(long_range2)
        long_range2 = F.dropout(long_range2, self.drop_rate, self.training)

        short_range2 = self.down_conv2(long_range2)

        long_range3 = self.encoder_stage3(short_range2) + short_range2
        long_range3 = self.batch_norm3(long_range3)
        long_range3 = F.dropout(long_range3, self.drop_rate, self.training)

        short_range3 = self.down_conv3(long_range3)

        long_range4 = self.encoder_stage4(short_range3) + short_range3
        long_range4 = self.batch_norm4(long_range4)
        long_range4 = F.dropout(long_range4, self.drop_rate, self.training)

        short_range4 = self.down_conv4(long_range4)

        return short_range4, long_range1, long_range2,long_range3,long_range4
#for weights in 3D_1*w1+3D_2 = GT
class Sample(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.param = nn.Parameter(torch.ones(1,2,3, dtype=torch.float32))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self.param * input

"""
if __name__ == '__main__':
    net = ResUNet()
    down = ResUNet_Down()
    keys = set(down.state_dict().keys())
    down.load_state_dict({k:v for k,v in net.state_dict().items() if k in keys})
"""