from torch.utils.data import Dataset
import os


class BootDataset(Dataset):
    def __init__(self,path):
        super(BootDataset,self).__init__()
        self.dirs=os.listdir(path)
        self.items=[]
        for dir in self.dirs:
            dir_path=os.path.join(path,dir)
            self.items+=[os.path.join(dir_path,img_path) for img_path in os.listdir(dir_path)]


    def __getitem__(self, idx):

        return self