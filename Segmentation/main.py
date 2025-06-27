from Loop import Train_model
from Model import Unet
from Dataset import Boot_Segmentation_Dataset
import yaml
from torch.optim import AdamW
from torch.nn import BCELoss
import torch
from torch.utils.data import DataLoader
option_path=fr'D:/Code/KD_Summer_Work/config.yml'
with open(option_path,'r') as file_option:
    option=yaml.safe_load(file_option)

dataset=Boot_Segmentation_Dataset(option['Segmentation']['img_path'],option['Segmentation']['label_path'])

dataloader=DataLoader(dataset=dataset,batch_size=16,drop_last=False,shuffle=True)

model=Unet(3,32)
loss_fn=BCELoss()
optimizer=AdamW(model.parameters(),lr=0.001)

try:
    weights_dict=torch.load(option['Segmentation']['weights_path'],weights_only=True)
    model.load_state_dict(weights_dict)
except:
    print('Весов нет, инициализируем новые')

Train_model(model=model,dataloader=dataloader,loss_func=loss_fn,optimizer=optimizer,epochs=2)

