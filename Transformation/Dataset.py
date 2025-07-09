import PIL.Image
from torch.utils.data import Dataset
import os
import PIL
from torchvision import transforms

class BootDataset(Dataset):
    def __init__(self,path,transformation=None):
        super(BootDataset,self).__init__()
        self.dirs=os.listdir(path)
        self.start_items=[]
        self.done_items=[]
        for dir in self.dirs:
            dir_path=os.path.join(path,dir)
            for img in os.listdir(dir_path):  
                try:
                    if os.path.join(dir_path,f'{img[0:14]}.JPG') not in self.start_items:
                        self.start_items.append(os.path.join(dir_path,f'{img[0:14]}.JPG'))
                        self.done_items.append(os.path.join(dir_path,f'{img[0:14]}_done.JPG'))
                except:
                    print(f'не найдена пара для {img}')

            
        self.transformation=transformation
        if self.transformation==None:
            self.transformation=transforms.Compose([
            transforms.Resize((762,1100)),
            transforms.ToTensor()
            ])
            
    def __len__(self):
        return len(self.start_items)
            


                #self.start_items+=[os.path.join(dir_path,img_path) for img_path in os.listdir(dir_path) if not 'done' in img_path ]
                #self.done_items+=[os.path.join(dir_path,img_path) for img_path in os.listdir(dir_path) if 'done' in img_path ]


    def __getitem__(self, idx):

        start_img=self.transformation(PIL.Image.open(self.start_items[idx]))
        done_img=self.transformation(PIL.Image.open(self.done_items[idx]))
        
        return {
            'start_img':start_img,
            'done_img':done_img,
        }
