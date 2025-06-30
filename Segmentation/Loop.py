from tqdm import tqdm
import yaml
import torch
#from progressbar import AdaptiveETA, ProgressBar, Timer
from torch import nn

option_path=fr'D:/Code/KD_Summer_Work/config.yml'
with open(option_path,'r') as file_option:
    option=yaml.safe_load(file_option)

#bar = ProgressBar(widgets=widgets, max_value=100).start()
def Train_model(model,dataloader,loss_func,optimizer,epochs):
    #loss_item=0#костыль
    sigm=nn.Sigmoid()
    for epoch in range(epochs):
        for batch in (pbar:=tqdm(dataloader)):
            optimizer.zero_grad()
            pred=sigm(model(batch['img']))

            loss=loss_func(pred,batch['label'])
            loss_item=loss.item()

            loss.backward()
            optimizer.step()
            pbar.set_description(f'loss: {loss_item}')

        try:
            torch.save(model.state_dict(),option['Segmentation']['weights_path'])
        except:
            print('ошибка сохранения весов')
