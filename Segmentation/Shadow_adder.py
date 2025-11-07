import torch
from torch import nn
from torch.nn import functional as F


def dilate(mask,R=20):
    kernel_size=2*R +1
    dilitation_kernel=torch.ones(1,1,kernel_size,kernel_size)
    new_mask=F.conv2d(mask.float(),dilitation_kernel,padding=R)
    new_mask=new_mask>0
    return new_mask

def cut_boot_from_photo(img,mask):
    sig=nn.Sigmoid()
    real_mask = (sig(mask)>(sig(mask).max()+sig(mask).min())/2) 
    new_img=real_mask.float()*img
    non_mask = (sig(mask)<(sig(mask).max()+sig(mask).min())/2).float()
    new_img+=non_mask    
    dilated_mask=dilate(real_mask,R=50)
    dilated_background=(dilated_mask<0.5).float()*img
    dilated_around_background=(dilated_mask.float()-real_mask.float())*img
    dilated_around_non_background=(dilated_mask.float()-real_mask.float())*new_img
    mn_mx=0.9
    koef=0.3
    new_shadow=koef*((dilated_around_background<mn_mx).float()*dilated_around_background)
    new_img_with_shadow=new_img-koef*((dilated_around_background<mn_mx).float()*dilated_around_non_background)+koef*((dilated_around_background<mn_mx).float()*dilated_around_background)
    return new_img_with_shadow