from torch.utils.data import Dataset
from PIL import Image
import os

class Boot_Segmentation_Dataset(Dataset):
    def __init__(self,img_path,label_path):
        super(Boot_Segmentation_Dataset,self).__init__()

        self.all_items=[os.path.join(img_path,img) for img in os.listdir(img_path)]
        self.all_labels=[os.path.join(label_path,label) for label in os.listdir(label_path)]
    def __len__(self):
        return len(self.all_items)
    def __getitem__(self,idx):
        img=Image.open(self.all_items[idx])
        label=Image.open(self.all_labels[idx])

        return {
            'img':img,
            'label':label
        }