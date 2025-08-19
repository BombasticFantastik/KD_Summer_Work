from Loop import Train_model
from Model import Unet
from Dataset import Boot_Segmentation_Dataset
import yaml
from Batching import img2batch,batch2img,img4batch,batch4img
from torch.optim import AdamW
from torch.nn import BCELoss
import torch
import torch.utils.data as data_utils
#indices = torch.arange(500)

import segmentation_models_pytorch as smp

import segmen
from torch.utils.data import DataLoader
option_path=fr'/home/artemybombastic/MyGit/KD_Summer_Work/config.yml'
device='cuda' if torch.cuda.is_available() else 'cpu'
#device='cuda'
print(device)
with open(option_path,'r') as file_option:
    option=yaml.safe_load(file_option)



dataset=Boot_Segmentation_Dataset(option['Segmentation']['img_path'],option['Segmentation']['label_path'],)
dataloader=DataLoader(dataset=dataset,batch_size=4,drop_last=False,shuffle=True)
assert dataset.all_items[0].split('/')[-1][:-4]==dataset.all_labels[0].split('/')[-1][:-4]
model=smp.Unet('mobilenet_v2', encoder_weights='imagenet', classes=1, activation=None, encoder_depth=5, decoder_channels=[256, 128, 64, 16, 4])
try:
    weights_dict=torch.load(f'/home/artemybombastic/MyGit/KD_Data/SegmData/unet_model_{device}.pth',weights_only=True)
    model.load_state_dict(weights_dict)
except:
    print('Весов нет, инициализируем новые')

loss_fn=BCELoss()
optimizer=AdamW(model.parameters(),lr=0.001)