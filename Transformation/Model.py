from torch import nn
from torchvision import models
from torchvision.models import ResNet50_Weights

class Degree_Net(nn.Module):
    def __init__(self,input_size,hidden_size):
        super(Degree_Net,self).__init__()
    
        self.lay0=nn.Sequential(
            nn.Conv2d(input_size,hidden_size,3,padding=1),
            nn.BatchNorm2d(hidden_size),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2,2)
        )
        
        self.lay1=nn.Sequential(
            nn.Conv2d(hidden_size,hidden_size*2,3,padding=1),
        
            nn.BatchNorm2d(hidden_size*2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2,2)
        )
        
        self.lay2=nn.Sequential(
            nn.Conv2d(hidden_size*2,hidden_size*4,3,padding=1),
            nn.BatchNorm2d(hidden_size*4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2,2)
        )
        
        self.lay3=nn.Sequential(
            nn.Conv2d(hidden_size*4,hidden_size*8,3,padding=1),
            nn.BatchNorm2d(hidden_size*8),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2,2)
        )
        
        self.linear_lay=nn.Sequential(
            nn.Flatten(),
            nn.Linear(3214080,128),
            nn.ReLU(),
            nn.Linear(128,1)
        )

    def forward(self,x):
        print(x.shape)
        x0=self.lay0(x)
        print(x0.shape)
    
        x1=self.lay1(x0)
          
        print(x1.shape)
    
        x2=self.lay2(x1)
        print(x2.shape)
    
        x3=self.lay3(x2)
        print(x3.shape)
        
    
        final_x=self.linear_lay(x2)
        print(final_x.shape)
    
        return final_x

class Transfered_Resnet50(nn.Module):
    def __init__(self):
        super().__init__()
        self.core=models.resnet50(weights=ResNet50_Weights.DEFAULT)

        lin_shape=self.core.fc.in_features
        self.core.fc=nn.Sequential(
            nn.Linear(lin_shape,512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512,1)
            
        )
            
    def forward(self,x):
        x=self.core(x)
        return x

        

