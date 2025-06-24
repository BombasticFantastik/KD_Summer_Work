import PIL.Image
from torch.utils.data import Dataset
import os
import PIL

class BootDataset(Dataset):
    def __init__(self,path):
        super(BootDataset,self).__init__()
        self.dirs=os.listdir(path)
        self.start_items=[]
        self.done_items=[]
        for dir in self.dirs:
            dir_path=os.path.join(path,dir)
            self.start_items+=[os.path.join(dir_path,img_path) for img_path in os.listdir(dir_path) if not 'done' in img_path ]
            self.done_items+=[os.path.join(dir_path,img_path) for img_path in os.listdir(dir_path) if 'done' in img_path ]


    def __getitem__(self, idx):

        img=PIL.Image.open(self.start_items[idx])
        label=PIL.Image.open(self.done_items[idx])
        
        return {
            'start_img':img,
            'done_img':label,
        }