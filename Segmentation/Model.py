from torch.nn import Module
from torch import nn
import torch

class DoubleConv(Module):
    def __init__(self,input_size,output_size):
        super(DoubleConv,self).__init__()

        self.conv_lay=nn.Sequential(
            nn.Conv2d(input_size,output_size,kernel_size=3,padding=1),
            nn.BatchNorm2d(output_size),
            nn.ReLU(inplace=True),
            nn.Conv2d(output_size,output_size,kernel_size=3,padding=1),
            nn.BatchNorm2d(output_size),
            nn.ReLU(inplace=True)
        )        
    def forward(self,x):
        conved_x=self.conv_lay(x)

class DownConv(Module):
    def __init__(self,input_size,output_size):
        super(DownConv,self).__init__()

        self.down_lay=DoubleConv(input_size=input_size,output_size=output_size)
        self.down_sample=nn.MaxPool2d(2)

    def forward(self,x):
        back_conved_x=self.down_lay(x)
        upsampled_x=self.down_sample(back_conved_x)
        return upsampled_x

class UpConv(Module):
    def __init__(self,input_size,output_size):
        super(UpConv,self).__init__()

        self.up_samlpe=nn.ConvTranspose2d(input_size,output_size,kernel_size=2, stride=2)
        self.double_conv=nn.DoubleConv(input_size+output_size,output_size)

    def forward(self,x,skiped_x):
        x=self.up_samlpe(x)
        cat_x=torch.cat([x,skiped_x],dim=1)
        return self.double_conv(cat_x)

class Unet(Module):
    def __init__(self,input_size,hidden_dim):
        super(Unet,self).__init__()

        #down
        self.down_conv0=DownConv(input_size,hidden_dim)
        self.down_conv1=DownConv(hidden_dim,hidden_dim*2)
        self.down_conv2=DownConv(hidden_dim*2,hidden_dim*4)
        self.down_conv3=DownConv(hidden_dim*4,hidden_dim*8)

        #bottleneck
        self.bottleneck=DoubleConv(hidden_dim*8,hidden_dim*16)

        #up
        self.up_conv3=UpConv(hidden_dim*16,hidden_dim*8)
        self.up_conv2=UpConv(hidden_dim*8,hidden_dim*4)
        self.up_conv1=UpConv(hidden_dim*4,hidden_dim*2)
        self.up_conv0=UpConv(hidden_dim*2,hidden_dim)

        #finlin
        self.last_conv=nn.Conv2d(hidden_dim,output_size,kernel_size=1)


        

    def forward(self,x):

        




