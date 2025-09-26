from tqdm import tqdm
import yaml
import torch
import torch.utils.data as data_utils

from torchvision.transforms.functional import adjust_contrast,adjust_sharpness

#from progressbar import AdaptiveETA, ProgressBar, Timer
from torch import nn

option_path=fr'/home/artemybombastic/MyGit/KD_Summer_Work/config.yml'
with open(option_path,'r') as file_option:
    option=yaml.safe_load(file_option)

def Train_model(model,dataloader,loss_func,optimizer,device):
    #loss_item=0#костыль
    model=model.to(device)
    sigm=nn.Sigmoid()
    for batch in (pbar:=tqdm(dataloader)):
        optimizer.zero_grad()
        batch['img']=adjust_sharpness(adjust_contrast(batch['img'],2),2)
        pred=sigm(model(batch['img'].to(device)))
        loss=loss_func(pred,batch['label'].to(device))
        loss_item=loss.item()
        loss.backward()
        optimizer.step()
        pbar.set_description(f'loss: {loss_item}')
        try:
            torch.save(model.state_dict(),f'/home/artemybombastic/MyGit/KD_Data/SegmData/unet_model_{device}.pth')
        except:
            print('ошибка сохранения весов')