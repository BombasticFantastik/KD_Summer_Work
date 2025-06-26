from torch.nn import Module
from torch import nn
import torch
class Unet(Module):
    def __init__(self,input_size,hidden_dim):
        super(Unet,self).__init__()

        self.ups=nn.Upsample(scale_factor=2,mode='bilinear')

        self.lay0=nn.Sequential(
            nn.Conv2d(input_size,hidden_dim,3,padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(),
            nn.Conv2d(hidden_dim,hidden_dim,3,padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.lay1=nn.Sequential(
            nn.Conv2d(hidden_dim,hidden_dim*2,3,padding=1),
            nn.BatchNorm2d(hidden_dim*2),
            nn.ReLU(),
            nn.Conv2d(hidden_dim*2,hidden_dim*2,3,padding=1),
            nn.BatchNorm2d(hidden_dim*2),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.lay2=nn.Sequential(
            nn.Conv2d(hidden_dim*2,hidden_dim*4,3,padding=1),
            nn.BatchNorm2d(hidden_dim*4),
            nn.ReLU(),
            nn.Conv2d(hidden_dim*4,hidden_dim*4,3,padding=1),
            nn.BatchNorm2d(hidden_dim*4),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.lay3=nn.Sequential(
            nn.Conv2d(hidden_dim*4,hidden_dim*8,3,padding=1),
            nn.BatchNorm2d(hidden_dim*8),
            nn.ReLU(),
            nn.Conv2d(hidden_dim*8,hidden_dim*8,3,padding=1),
            nn.BatchNorm2d(hidden_dim*8),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.lay4=nn.Sequential(
            nn.Conv2d(hidden_dim*8,hidden_dim*16,3,padding=1),
            nn.BatchNorm2d(hidden_dim*16),
            nn.ReLU(),
            nn.Conv2d(hidden_dim*16,hidden_dim*16,3,padding=1),
            nn.BatchNorm2d(hidden_dim*16),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.bottleneck=nn.Sequential(
            nn.Conv2d(hidden_dim*16,hidden_dim*32,3,padding=1),
            nn.BatchNorm2d(hidden_dim*32),
            nn.ReLU(),
            nn.Conv2d(hidden_dim*32,hidden_dim*16,3,padding=1),
            nn.BatchNorm2d(hidden_dim*16),
            nn.ReLU(),

        )

        self.dec_lay0=nn.Sequential(
            nn.Conv2d(hidden_dim*32,hidden_dim*16,3,padding=1),
            nn.BatchNorm2d(hidden_dim*16),
            nn.ReLU(),
            nn.Conv2d(hidden_dim*16,hidden_dim*16,3,padding=1),
            nn.BatchNorm2d(hidden_dim*16),
            nn.ReLU()
        )

        self.dec_lay1=nn.Sequential(
            nn.Conv2d(hidden_dim*16,hidden_dim*8,3,padding=1),
            nn.BatchNorm2d(hidden_dim*8),
            nn.ReLU(),
            nn.Conv2d(hidden_dim*8,hidden_dim*8,3,padding=1),
            nn.BatchNorm2d(hidden_dim*8),
            nn.ReLU()
        )

        self.dec_lay2=nn.Sequential(
            nn.Conv2d(hidden_dim*8,hidden_dim*4,3,padding=1),
            nn.BatchNorm2d(hidden_dim*4),
            nn.ReLU(),
            nn.Conv2d(hidden_dim*4,hidden_dim*4,3,padding=1),
            nn.BatchNorm2d(hidden_dim*4),
            nn.ReLU()
        )

        self.dec_lay3=nn.Sequential(
            nn.Conv2d(hidden_dim*4,hidden_dim*2,3,padding=1),
            nn.BatchNorm2d(hidden_dim*2),
            nn.ReLU(),
            nn.Conv2d(hidden_dim*2,hidden_dim*2,3,padding=1),
            nn.BatchNorm2d(hidden_dim*2),
            nn.ReLU()
        )

        self.dec_lay4=nn.Sequential(
            nn.Conv2d(input_size*2,hidden_dim,3,padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(),
            nn.Conv2d(hidden_dim,hidden_dim,3,padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU()
        )

    def forward(self,x):

        #encoder
        x0=self.lay0(x)
        x1=self.lay1(x0)
        x2=self.lay2(x1)
        x3=self.lay3(x2)
        x4=self.lay4(x3)

        #bottleneck
        midl_x=self.bottleneck(x4)

        #decoder
        xdec0=self.dec_lay0(self.ups(torch.cat((midl_x,x4),dim=1)))
        xdec1=self.dec_lay1(self.ups(torch.cat((xdec0,x3),dim=1)))
        xdec2=self.dec_lay2(self.ups(torch.cat((xdec1,x2),dim=1)))
        xdec3=self.dec_lay3(self.ups(torch.cat((xdec2,x1),dim=1)))
        xdec4=self.dec_lay4(self.ups(torch.cat((xdec3,x0),dim=1)))

        return xdec4
