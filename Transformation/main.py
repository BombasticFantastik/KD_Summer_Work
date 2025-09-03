import yaml
from Dataset import Boot_Rotate_Dataset
import os
from Model import Transfered_Resnet50
import torch
from torch.utils.data import DataLoader
from torch import nn
# option_path=fr'D:/Code/KD_Summer_Work/config.yml' ---------------------подредачить
# with open(option_path,'r') as file_option:
#     print(file_option)
#     option=yaml.safe_load(file_option)


degr_dataset=Boot_Rotate_Dataset('/home/artemybombastic/MyGit/KD_Data/TransformData')#---------убрать прямую ссылку
degr_dataloader=DataLoader(degr_dataset,batch_size=4,shuffle=True,drop_last=True)
resnet_model=Transfered_Resnet50()#loss:93636.875,30000,63915.5
loss_func=nn.MSELoss()
optimizer=torch.optim.AdamW(resnet_model.parameters())
device='cpu'