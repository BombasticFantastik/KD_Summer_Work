import torch
from torchvision import transforms
def img2batch(img):
    trans=transforms.functional.crop

    #assert img.shape(1)==img.shape(2) 
    #assert img.shape(2)/2==img.shape(2)//2 

    step=int(img.size(2)/2)
    

    img_batch=[
        trans(img,0,0,step,step),
        trans(img,0,step,step,step),
        trans(img,step,0,step,step),
        trans(img,step,step,step,step)
    ]
    return torch.stack(img_batch)


def batch2img(batch):
    




