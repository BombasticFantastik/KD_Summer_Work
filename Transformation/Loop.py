import torch
from tqdm import tqdm
from torch import nn


# def accuracy(x,pred):
#     result=torch.abs(x-pred)
#     return result().sum()


def Train_degr_model(model,dataloader,loss_func,optimizer,device):
    #loss_item=0#костыль
    model=model.to(device)
    for batch in (pbar:=tqdm(dataloader)):
        optimizer.zero_grad()
        pred=model(batch['img'].to(device))
        #print(pred,batch['label'])
        loss=loss_func(pred,batch['label'].to(device))
        loss_item=loss.item()
        loss.backward()
        optimizer.step()
        pbar.set_description(f'loss: {loss_item}')
        try:
            torch.save(model.state_dict(),f'/home/artemybombastic/MyGit/KD_Data/degr_net_{device}.pth')
        except:
            print('ошибка сохранения весов')