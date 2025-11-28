import os
import torch
import yaml
from torch.utils.data import DataLoader
from torch import nn
from PIL import Image
from Dataset import Boot_Rotate_Dataset
from Model import Transfered_Resnet50
from Loop import Train_degr_model

option_path=fr'/home/artemybombastic/MyGit/KD_Summer_Work/config.yml'
with open(option_path,'r') as file_option:
    option=yaml.safe_load(file_option)

degr_dataset=Boot_Rotate_Dataset(option['Trans']['data_path'])#---------убрать прямую ссылку
degr_dataloader=DataLoader(degr_dataset,batch_size=4,shuffle=True,drop_last=True)

resnet_model=Transfered_Resnet50()

if f'degr_net_{option['device']}.pth' in os.listdir('/home/artemybombastic/MyGit/KD_Data/'):
    weights_dict=torch.load(f'/home/artemybombastic/MyGit/KD_Data/degr_net_{option['device']}.pth',weights_only=True)
    resnet_model.load_state_dict(weights_dict)
    print('yes')

loss_func=nn.MSELoss()
optimizer=torch.optim.AdamW(resnet_model.parameters())


Train_degr_model(model=resnet_model,dataloader=degr_dataloader,loss_func=loss_func,optimizer=optimizer,device=option['device'])