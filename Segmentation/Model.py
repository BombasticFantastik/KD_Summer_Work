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
        return conved_x

class DownConv(Module):
    def __init__(self,input_size,output_size):
        super(DownConv,self).__init__()

        self.down_lay=DoubleConv(input_size=input_size,output_size=output_size)
        self.down_sample=nn.MaxPool2d(2)

    def forward(self,x):

        back_conved_x=self.down_lay(x)
        
        upsampled_x=self.down_sample(back_conved_x)
        return (upsampled_x,back_conved_x)

class UpConv(Module):
    def __init__(self,input_size,output_size):
        super(UpConv,self).__init__()

        self.up_samlpe=nn.ConvTranspose2d(input_size,output_size,kernel_size=2, stride=2)
        self.double_conv=DoubleConv(input_size+output_size,output_size)

    def forward(self,x,skiped_x):
        print(x.shape)
        x=self.up_samlpe(x)
        print(x.shape,skiped_x.shape)
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
        self.last_conv=nn.Conv2d(hidden_dim,1,kernel_size=1)

    def forward(self,x):

        print(x.shape)

        x,skip0=self.down_conv0(x)
        print(x.shape,skip0.shape)

        x,skip1=self.down_conv1(x)
        print(x.shape,skip1.shape)

        x,skip2=self.down_conv2(x)
        print(x.shape,skip2.shape)

        x,skip3=self.down_conv3(x)

        print('_________')
        x=self.bottleneck(x)
        print(x.shape,skip3.shape)

        x=self.up_conv3(x,skip3)
        print(x.shape)
        x=self.up_conv2(x,skip2)
        print(x.shape)
        x=self.up_conv1(x,skip1)
        print(x.shape)
        x=self.up_conv0(x,skip0)
        print(x.shape)
        x=self.last_conv(x)
        print(x.shape)
        return x



        




