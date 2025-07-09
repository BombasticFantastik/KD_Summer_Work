from tqdm import tqdm
import yaml
import torch
import torch.utils.data as data_utils

#from progressbar import AdaptiveETA, ProgressBar, Timer
from torch import nn

option_path=fr'C:\Code\KD_PRACT\KD_Summer_Work\config.yml'
with open(option_path,'r') as file_option:
    option=yaml.safe_load(file_option)

#bar = ProgressBar(widgets=widgets, max_value=100).start()
# def Train_model(model,dataloader,loss_func,optimizer,epochs,device,batch_func=None,revevrse_batch_func=None):
#     #loss_item=0#костыль
#     model=model.to(device)
#     sigm=nn.Sigmoid()
#     for epoch in range(epochs):
#         for batch in (pbar:=tqdm(dataloader)):
#             optimizer.zero_grad()
#             if batch_func and revevrse_batch_func:
#                 real_batch=batch_func(batch['img'][0].to(device),device)
#             #     real_batch=torch.Tensor()
#             #     for img in batch['img']:
#             #         real_batch=torch.cat((real_batch,batch_func(img)),dim=0)
#             #         print(real_batch.shape)
#             # else:
#             #     real_batch=batch[img]

#             pred=sigm(model(real_batch.to(device)))
#             # if batch_func and revevrse_batch_func:
#             #     real_batch=torch.Tensor()
#             #     for img in batch['img']:
#             #         real_batch=torch.cat((real_batch,batch_func(img)),dim=0)
#             # else:
#             #     real_batch=batch[img]

            

#             pred=revevrse_batch_func(pred,device)
#             pred=pred.unsqueeze(0)
#             loss=loss_func(pred,batch['label'].to(device))
#             loss_item=loss.item()
#             loss.backward()
#             optimizer.step()
#             pbar.set_description(f'loss: {loss_item}')

#         try:
#             torch.save(model.state_dict(),option['Segmentation']['weights_path'])
#         except:
#             print('ошибка сохранения весов')

def Train_model(model,dataloader,loss_func,optimizer,epochs,device):
    #loss_item=0#костыль
    model=model.to(device)
    sigm=nn.Sigmoid()
    for epoch in range(epochs):
        for batch in (pbar:=tqdm(dataloader)):
            optimizer.zero_grad()
            pred=sigm(model(batch['img'].to(device)))
        
            loss=loss_func(pred,batch['label'].to(device))
            loss_item=loss.item()
            loss.backward()
            optimizer.step()
            pbar.set_description(f'loss: {loss_item}')

            try:
                torch.save(model.state_dict(),option['Segmentation']['weights_path'])
            except:
                print('ошибка сохранения весов')
