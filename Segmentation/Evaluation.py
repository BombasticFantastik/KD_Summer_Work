from matplotlib import pyplot as plt
from torch import nn
def Evaluate(model,dataloader):
    #tahn=nn.Sigmoid()
    for batch in dataloader:

        pred=model(batch['img'])
        fig = plt.figure(figsize=(25, 15))
        for i in range(1,(batch['img'].size(0)-1),3):
            plt.subplot(16, 3, i)  
            plt.imshow(pred[i-1].permute(2,1,0).detach().numpy())  
            plt.axis('off')  
            plt.title("pred") 
            
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