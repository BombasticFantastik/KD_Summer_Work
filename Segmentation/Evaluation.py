from matplotlib import pyplot as plt
from torch import nn
def Evaluate(model,dataloader,device):
    sigm=nn.Sigmoid()
    for batch in dataloader:

        pred=sigm(model(batch['img'].to(device)))
        fig = plt.figure(figsize=(25, 15))
        for i in range(1,(batch['img'].size(0)-1),3):
            plt.subplot(16, 3, i)  
            plt.imshow(pred[i-1].cpu().permute(2,1,0).detach().numpy())  
            plt.axis('off')  
            plt.title("pred") 
            print(1)
            
            plt.subplot(16, 3, i+1)  
            plt.imshow(batch['img'][i-1].permute(1,2,0).detach().numpy())  
            plt.axis('off')  
            plt.title("img") 


            plt.subplot(16, 3, i+2)  
            plt.imshow(batch['label'][i-1].permute(1,2,0).detach().numpy())  
            plt.axis('off')  
            plt.title("label") 

        plt.show()
        break