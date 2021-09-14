import os
import os.path
import cv2
import numpy as np
from utils import transform
from torch.utils.data import Dataset,DataLoader
from utils.utils import colorization
from utils import palette
import PIL
from PIL import Image
from torchvision.utils import make_grid
from torchvision import transforms
import torch


class VOC_Dataset(Dataset):
    def __init__(self, split='train_aug', data_root=None,  transform=None):
        self.split = split
        self.data_root=data_root
        self.nbr_classes=21
        self.data_list=self.__make_list(data_root,split)
        self.transform = transform  ## Operations for data augmentation
        self.MEAN = [0.45734706, 0.43338275, 0.40058118]
        self.STD = [0.23965294, 0.23532275, 0.2398498]
        self.palette=palette.get_voc_palette(21)
    def __make_list(self,data_root,split):

        assert split in ['train_aug','val_aug','test']
        name_list=[]
        text_file=os.path.join(data_root,"ImageSets/Segmentation",split+'.txt')
        line_list=open(text_file).readlines()
        #print(len(line_list))
        print("Totally have {} samples in {} set.".format(len(line_list),str(split)))
        for line in line_list:
            line = line.strip()
            line_split = line.split(' ')
            input_name = self.data_root+line_split[0]

            if split in ['train_aug','val_aug']:
                label_name = self.data_root+line_split[1]
            else:
                label_name=input_name
            name_item=(input_name,label_name)
            name_list.append(name_item)
        return name_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        input_path,label_path=self.data_list[index]
        input=Image.open(input_path)
        label=Image.open(label_path)
        if self.transform is not None:
            sample=self.transform({'img':input,'label':label})
        input,label=sample['img'],sample['label']
        #print(label.shape)
        class_label=self.__make_onehot(label)
        return input, label,class_label

    def __make_onehot(self, label, ignore=255):
        labels=label.detach()
        class_label = torch.zeros((self.nbr_classes, label.shape[0], label.shape[1]))
        labels[labels== ignore] = 0
        class_indices = list(np.unique(labels))
        for i in class_indices:
            value = int(i)
            class_label[value][labels == value] = 1
        return class_label.float()


if __name__ == '__main__':
    import argparse
    parser=argparse.ArgumentParser()
    args=parser.parse_args()
    trans=transform.Compose([transform.HorizontalFilp(),
                             transform.Crop(base_size=513,crop_height=513,crop_width=513,type='central'),
                             transform.GaussianBlur(),
                             transform.ToTensor(),
                             transform.Normalize(std = [0.17613647, 0.18099176, 0.17772235],mean = [0.28689529, 0.32513294, 0.28389176])
                             ])

    split='train_aug'
    data_root="D:\\datasets\\VOC\\VOCdevkit\\VOC2012"
    data_list="D:\\datasets\\VOC\\VOCdevkit\\VOC2012\\ImageSets\\Segmentation\\train_aug.txt"

    Voc_set=VOC_Dataset(split=split,data_root=data_root,transform=trans)
    col_map=Voc_set.palette
    dataloader = DataLoader(Voc_set, batch_size=10, shuffle=True, num_workers=0)

    val_list=[]
    for ii, sample in enumerate(dataloader):
        gt = sample[1][0]
        if (len(torch.unique(gt))>3):
            if len(val_list)<15:

                gt=colorization(gt.numpy(),col_map)

                img=sample[0][0]
                img=transform.DeNormalize(std = [0.17613647, 0.18099176, 0.17772235],mean = [0.28689529, 0.32513294, 0.28389176])(img)
                gt=transforms.ToTensor()(gt.convert('RGB'))
                val_list.extend([img,gt])
            else:
                break
    val_list=torch.stack(val_list,0)
    val_list=make_grid(val_list,nrow=2,padding=5)
    val_list=transforms.ToPILImage()(val_list)
    val_list.show()




