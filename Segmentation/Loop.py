#from tqdm import tqdm
from rich.progress import track
def Train_model(model,dataloader,loss_func,optimizer,epochs):
    for epoch in range(epochs):
        for batch in track(dataloader):
            optimizer.zero_grad()

            pred=model(batch['img'])

            loss=loss_func(pred,batch['label'])
            loss.item=loss.item

            loss.backward()
            optimizer.step()
            print(1)
