"""
This code is referenced from https://github.com/assassint2017/MICCAI-LITS2017
"""
from __future__ import print_function
from __future__ import absolute_import
from __future__ import division
from turtle import forward
import torch
import torch.nn as nn
import torch.nn.functional as F
from lib.model.base_model import Base_Model
import lib.model.nets.factory as factory




class ResUNet_up(nn.Module):
    #dim=16
    def __init__(self,in_channel=256,out_channel=1,training=True):
        super(ResUNet_up,self).__init__()
        self.w = torch.nn.Parameter(torch.tensor([0.5,0.5]))
        self.drop_rate = 0.2
        self.training = training
        #dim = int(in_channel*(dim**3))
        #print("fiem")
        #print(dim)
        self.calibration = nn.Linear(2,2)

        self.down_conv = nn.Sequential(
            nn.Conv3d(in_channel,int(in_channel/2),kernel_size=3,stride=1,padding=1),
            nn.PReLU(int(in_channel/2)),
            nn.MaxPool3d(2,stride=2, padding=0),
            nn.BatchNorm3d(int(in_channel/2))
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

        self.decoder_stage2 = nn.Sequential(
            nn.Conv3d(64, 64, 3, 1, padding=1),
            nn.PReLU(64),

            nn.Conv3d(64, 64, 3, 1, padding=1),
            nn.PReLU(64),

            nn.Conv3d(64, 64, 3, 1, padding=1),
            nn.PReLU(64),
        )

        self.decoder_stage3 = nn.Sequential(
            nn.Conv3d(32, 32, 3, 1, padding=1),
            nn.PReLU(32),

            nn.Conv3d(32, 32, 3, 1, padding=1),
            nn.PReLU(32),

            nn.Conv3d(32, 32, 3, 1, padding=1),
            nn.PReLU(32),
        )
        self.decoder_stage4 = nn.Sequential(
            nn.Conv3d(16, 16, 3, 1, padding=1),
            nn.PReLU(16),

            nn.Conv3d(16, 16, 3, 1, padding=1),
            nn.PReLU(16),
        )

        # 最后大尺度下的映射（256*256），下面的尺度依次递减
        self.map4 = nn.Sequential(
            nn.Conv3d(16, out_channel, 1, 1),
            nn.Upsample(scale_factor=(2, 2, 2), mode='trilinear', align_corners=False),
            #nn.Softmax(dim=1)
        )


        

        
    def forward(self,X,pred_G):
        X = self.down_conv(X)
        #X = F.softmax(X,dim=1)
       #X.size() = torch.Size([1, 128, 8, 8, 8])
        #print(X.size())
        short_range6 = self.up_conv2(X)
        #print(X.size())
        X = self.decoder_stage2(short_range6) + short_range6
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
        print(self.w.size())
        
        #w = self.calibration(self.w)
        w = F.softmax(self.w,dim=0)
        #print("calibr weights: {}".format(self.w))
        res = X*w[0] + pred_G * w[1]
        

        return res
        
        



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
    @property
    def name(self):
        return 'multiView_AutoEncoder'

    def forward(self, inputs):

        #print("input")
        #print(str(inputs.size()))

        long_range1 = self.encoder_stage1(inputs) + inputs

        short_range1 = self.down_conv1(long_range1)

        long_range2 = self.encoder_stage2(short_range1) + short_range1
        long_range2 = F.dropout(long_range2, self.drop_rate, self.training)

        short_range2 = self.down_conv2(long_range2)

        long_range3 = self.encoder_stage3(short_range2) + short_range2
        long_range3 = F.dropout(long_range3, self.drop_rate, self.training)

        short_range3 = self.down_conv3(long_range3)

        long_range4 = self.encoder_stage4(short_range3) + short_range3
        long_range4 = F.dropout(long_range4, self.drop_rate, self.training)

        short_range4 = self.down_conv4(long_range4)

        outputs = self.decoder_stage1(long_range4) + short_range4
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

        
    @property
    def name(self):
        return 'multiView_AutoEncoder'

    def forward(self, inputs):

        long_range1 = self.encoder_stage1(inputs) + inputs

        short_range1 = self.down_conv1(long_range1)

        long_range2 = self.encoder_stage2(short_range1) + short_range1
        long_range2 = F.dropout(long_range2, self.drop_rate, self.training)

        short_range2 = self.down_conv2(long_range2)

        long_range3 = self.encoder_stage3(short_range2) + short_range2
        long_range3 = F.dropout(long_range3, self.drop_rate, self.training)

        short_range3 = self.down_conv3(long_range3)

        long_range4 = self.encoder_stage4(short_range3) + short_range3
        long_range4 = F.dropout(long_range4, self.drop_rate, self.training)

        short_range4 = self.down_conv4(long_range4)

        return short_range4
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