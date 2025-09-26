from torch.utils.data import Dataset
from PIL import Image
import os
from torchvision import transforms

class Boot_Segmentation_Dataset(Dataset):
    def __init__(self,img_path,label_path,transformation=None,back_transformation=None):
        super(Boot_Segmentation_Dataset,self).__init__()
        self.transformation=transformation
        self.back_transformation=back_transformation
        self.all_items=[os.path.join(img_path,img) for img in os.listdir(img_path)]
        self.all_labels=[os.path.join(label_path,label) for label in os.listdir(label_path)]
        self.all_items.sort()
        self.all_labels.sort()

        if self.transformation==None:
            self.transformation=transforms.Compose([
            #transforms.RandomResizedCrop(size=(2000, 2000)),
            transforms.Resize((1024,512)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),

            transforms.ToTensor(),

            ])
        if self.back_transformation==None:
            self.back_transformation=transforms.Compose([
            #transforms.RandomResizedCrop(size=(2000, 2000)),
            transforms.Resize((512,1024)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),

            transforms.ToTensor()
            ])
    def __len__(self):
        return len(self.all_items)
    def __getitem__(self,idx):
        img=Image.open(self.all_items[idx])
        label=Image.open(self.all_labels[idx])




        tensor_img=self.transformation(img)
        tensor_label=self.transformation(label)


        if img.size==(1824,1216):
          tensor_img=self.back_transformation(img).permute(0,2,1)
          tensor_label=self.back_transformation(label).permute(0,2,1)


        return {
            'img':tensor_img,
            'label':tensor_label
        }