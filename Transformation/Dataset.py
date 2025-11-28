from PIL import Image
from torch.utils.data import Dataset
import os
import torch
from random import randint
import PIL
from torchvision import transforms

class Boot_Rotate_Dataset(Dataset):
  def __init__(self,path):
    super(Boot_Rotate_Dataset,self).__init__()

    dirs=[os.path.join(path,dir) for dir in os.listdir(path)]

    self.all_images=[]
    for dir in dirs:
      images=[os.path.join(dir,img) for img in os.listdir(dir) if 'done' in img]
      self.all_images+=images

    # self.trans=transforms.Compose([

    #     transforms.Resize((762,1100))
    # ])
    self.tensor_trans=transforms.Compose([

        transforms.Resize((762,1100)),
        transforms.ToTensor(),
        #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


  def __len__(self):
    return len(self.all_images)

  def __getitem__(self,idx):

    img=Image.open(self.all_images[idx])
    original_img=img

    degr=randint(-20,20)
    img=img.rotate(degr, expand=False, fillcolor=(255, 255, 255))
    # new_width,new_height=(610,932)

    # width,height=img.size
    # left=(width-new_width)//2
    # top=(height-new_height)//2
    # right=left+new_width
    # bottom=top+new_height
    # img=img.crop((left,top,right,bottom))
    tensor_img=self.tensor_trans(img)
    return {
        "img": tensor_img,
        "label":torch.FloatTensor([degr]),
        'real_img':self.tensor_trans(original_img)
        }