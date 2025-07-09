from torch.utils.data import Dataset
from PIL import Image
import os
from torchvision import transforms

class Boot_Segmentation_Dataset(Dataset):
    def __init__(self,img_path,label_path,transformation=None):
        super(Boot_Segmentation_Dataset,self).__init__()
        self.transformation=transformation
        self.all_items=[os.path.join(img_path,img) for img in os.listdir(img_path)]
        self.all_labels=[os.path.join(label_path,label) for label in os.listdir(label_path)]

        if self.transformation==None:
            self.transformation=transforms.Compose([
            transforms.Resize((762,1100)),
            transforms.ToTensor()
            ])
    def __len__(self):
        return len(self.all_items)
    def __getitem__(self,idx):
        img=self.transformation(Image.open(self.all_items[idx])) #.rotate(90,expand=True)
        label=self.transformation(Image.open(self.all_labels[idx]))

        return {
            'img':img,
            'label':label
        }