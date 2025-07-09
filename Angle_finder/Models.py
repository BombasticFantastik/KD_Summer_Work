import torch
from torch import nn 
from torch.nn import Module

class Angle_CNN_2D(Module):
    def __init__(self,in_chanels,hidden_dim):
        super(Angle_CNN_2D,self).__init__()

        self.lay0=nn.Sequential(
            nn.Conv2d(in_chanels,hidden_dim),
            nn.ReLU(),
            nn.Conv2d(hidden_dim,hidden_dim)
        )
        self.flat=nn.Flatten()
        self.fc=nn.Linear(hidden_dim*hidden_dim,1)

    def forward(self,x):
        print(x.shape)
        x=self.lay0(x)
        print(x.shape)
        x=self.flat(x)
        print(x.shape)
        x=self.fc(x)
        print(x.shape)
        return x

        