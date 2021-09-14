import random
import numpy as np
from PIL import Image, ImageOps, ImageFilter
from skimage.filters import gaussian
import torch
import math
import numbers
import random
import PIL

def make_one_hot(labels, classes):
    labels[labels == 255] = 0
    labels = labels.unsqueeze(0).unsqueeze(0)
    one_hot = torch.FloatTensor(labels.size()[0], classes, labels.size()[2], labels.size()[3]).zero_()
    target = one_hot.scatter_(1, labels.data, 1)[0]
    return target

class Compose(object):
    def __init__(self,aug_trans):
        self.trans=aug_trans

    def __call__(self,sample):
        for op in self.trans:
            sample=op(sample)
        return sample


class Resize(object):
    def __init__(self,size):
        self.size=(size,size)


    def __call__(self,sample):
        img=sample['img']
        label=sample['label']

        assert img.size==label.size

        img=img.resize(self.size,resample=Image.BILINEAR)
        label=label.resize(self.size,resample=Image.NEAREST)
        sample={
            'img':img,
            'label':label
        }
        return sample

class Scale(object):
    """
    scale the input image with a fixed scale factor.
    """
    def __init__(self,factor):
        self.scale_factor=factor

    def __call__(self,sample):
        img=sample['img']
        label=sample['label']
        assert img.size==label.size

        w,h=img.size
        target_w=int(self.scale_factor*w)
        target_h=int(self.scale_factor*h)

        img=img.resize((target_w,target_h),resample=Image.BILINEAR)
        label=label.resize((target_w,target_h),resample=Image.NEAREST)
        sample={
            'img':img,
            'label':label
        }
        return sample

class Crop(object):
    def __init__(self,base_size,crop_width,crop_height,type='rand',scale_min=0.5,scale_max=2.0):
        self.base_size=base_size  ## the long size of the original input
        self.crop_width=crop_width
        self.crop_height=crop_height
        self.scale_min=scale_min
        self.scale_max=scale_max
        if type=='central':
            self.crop=CentralCrop(crop_width,crop_height)
        else:
            self.crop=RandomCrop(crop_width,crop_height)

    def __call__(self, sample):
        img=sample['img']
        label=sample['label']

        w,h=img.size
        scale_factor=random.uniform(self.scale_min,self.scale_max)
        long_side=int(scale_factor*self.base_size)

        w,h=(long_side,int((1.0*long_side*h/w+0.5))) if w>h else (int((1.0*long_side*w/h+0.5)),long_side)


        ## main the ratio of the original image.
        img=img.resize((w,h),resample=Image.BILINEAR)
        label=label.resize((w,h),resample=Image.NEAREST)

        if(w>self.crop_width):
            pad_w=0
        else:
            pad_w=(self.crop_width-w)//2+1
        if(h>self.crop_height):
            pad_h=0
        else:
            pad_h=(self.crop_height-h)//2+1
        if pad_w or pad_h:
            border=(pad_w,pad_h,pad_w,pad_h)
            img=ImageOps.expand(img,border=border,fill=1)
            label=ImageOps.expand(label,border=border,fill=255)

        sample={
            'img':img,
            'label':label
        }
        return self.crop(sample)


class RandomCrop(object):
    def __init__(self,crop_width,crop_height):
        self.crop_width=crop_width
        self.crop_height=crop_height

    def __call__(self, sample):
        img=sample['img']
        label=sample['label']
        assert img.size==label.size
        w,h=img.size

        start_x=random.randint(0,w-self.crop_width)
        start_y=random.randint(0,h-self.crop_height)

        img=img.crop((start_x,start_y,start_x+self.crop_width,start_y+self.crop_height))
        label=label.crop((start_x,start_y,start_x+self.crop_width,start_y+self.crop_height))

        sample={
            'img':img,
            'label':label
        }
        return sample

class CentralCrop(object):
    def __init__(self,crop_width,crop_height):
        self.crop_width=crop_width
        self.crop_height=crop_height

    def __call__(self, sample):
        img=sample['img']
        label=sample['label']
        assert img.size==label.size
        w,h=img.size

        start_x=int((w-self.crop_width)/2.)
        start_y=int((h-self.crop_height)/2.)

        img=img.crop((start_x,start_y,start_x+self.crop_width,start_y+self.crop_height))
        label=label.crop((start_x,start_y,start_x+self.crop_width,start_y+self.crop_height))

        sample={
            'img':img,
            'label':label
        }
        return sample

class HorizontalFilp(object):
    def __call__(self,sample):
        img=sample['img']
        label=sample['label']
        if random.random()<0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            label = label.transpose(Image.FLIP_LEFT_RIGHT)

        sample = {
            'img': img,
            'label': label
        }
        return sample


class RandomRotation(object):
    def __init__(self,degree):
        self.degree=degree

    def __call__(self, sample):
        img = sample['img']
        label = sample['label']
        assert img.size == label.size

        rotate_degree=random.randint(-1*self.degree,self.degree)

        img=img.rotate(rotate_degree,Image.BILINEAR)
        label=label.rotate(rotate_degree,Image.NEAREST)

        sample = {
            'img': img,
            'label': label
        }
        return sample

class  GaussianBlur(object):
    def __call__(self, sample):
        img=sample['img']
        label=sample['label']
        assert img.size==label.size

        if random.random()<0.5:
            img=img.filter(ImageFilter.GaussianBlur(
                    radius=random.random()))

        sample = {
            'img': img,
            'label': label
        }
        return sample

class Normalize(object):
    def __init__(self,std,mean):
        self.std=std
        self.mean=mean

    def __call__(self, sample):
        img=sample['img']
        label=sample['label']

        for t, m, s in zip(img, self.mean, self.std):
            t.sub_(m).div_(s)
        sample={
            'img':img,
            'label':label
        }
        return sample

class DeNormalize(object):
    def __init__(self, std, mean):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        return tensor*255

class ToTensor(object):
    def __call__(self, sample):
        img=sample['img']
        label=sample['label']

        img=np.transpose(img,(2,0,1))
        label=np.asarray(label)
        img=torch.from_numpy(img).float()
        label=torch.from_numpy(label).long()

        sample={
            'img':img,
            'label':label
        }
        return sample

class df_viz(object):
    def __call__(self, img):
        img=255*(img-img.min())/(img.max()-img.min())
        #print(img.shape)
        return img.astype(np.uint8)

if __name__=='__main__':
    img=Image.open('C:\\Users\\user\\Desktop\\model\\2008_004016.jpg')
    img = img.filter(ImageFilter.GaussianBlur(
                radius=random.random()))
    img.show()
    # sample={
    #     'img':img,
    #     'label':img
    # }
    # img=Crop(base_size=800,crop_width=513,crop_height=513,type='central',)(sample)
    # img=img['img'].show()
