import torch
from torchvision import transforms
def img2batch(img):
    trans=transforms.functional.crop
    step=int(img.size(2)/2)
    img_batch=[
        trans(img,0,0,step,step),
        trans(img,0,step,step,step),
        trans(img,step,0,step,step),
        trans(img,step,step,step,step)
    ]
    return torch.stack(img_batch)

def batch2img(batch):


    first=torch.cat((batch[0],batch[1]),dim=2)
    second=torch.cat((batch[2],batch[3]),dim=2)
    img=torch.cat((first,second),dim=1)

    return img





