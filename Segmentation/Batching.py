import torch
from torchvision import transforms
def img2batch(img,device='cpu'):
    trans=transforms.functional.crop
    step=int(img.size(2)/2)
    img_batch=[
        trans(img,0,0,step,step),
        trans(img,0,step,step,step),
        trans(img,step,0,step,step),
        trans(img,step,step,step,step)
    ]
    return torch.stack(img_batch)

def batch2img(batch,device='cpu'):
    first=torch.cat((batch[0],batch[1]),dim=2)
    second=torch.cat((batch[2],batch[3]),dim=2)
    img=torch.cat((first,second),dim=1)
    return img

def batch4img(batch,device='cpu',part_count=4):
  hor_batch=torch.Tensor().to(device)
  j=0
  i=part_count
  for y in range(part_count):
    vert_batch=torch.Tensor().to(device)
    for x in range(j,i):
      vert_batch=torch.cat((vert_batch,batch[x]),dim=1)
      #print(vert_batch.shape)
    j+=4
    i+=4
    hor_batch=torch.cat((hor_batch,vert_batch),dim=2)
  return hor_batch


def img4batch(img,device='cpu',shape=(2048,2048),part_count=4):
  transform=transforms.functional.crop
  width=shape[0]
  height=shape[1]
  
  step=width/part_count

  img_batch=torch.Tensor().to(device)
  img_batch=[]
  for width_cnt in range(part_count):
    for height_cnt in range(part_count):
      batch_shape=[

          int(height_cnt*step),
          int(width_cnt*step),
          int(step),
          int(step)
      ]


      img_batch.append(transform(img,*batch_shape))
  return torch.stack(img_batch)










