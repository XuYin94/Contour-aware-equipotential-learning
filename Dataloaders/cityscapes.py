import os
import os.path
import cv2
import numpy as np
from utils import transform
from torch.utils.data import Dataset,DataLoader
from torchvision import transforms
from utils import palette
from utils.utils import colorization
from glob import glob
import PIL.Image as Image
import PIL
from torchvision.utils import make_grid
from torchvision import transforms
import torch
ignore_label = 255
ID_TO_TRAINID = {-1: ignore_label, 0: ignore_label, 1: ignore_label, 2: ignore_label,
                    3: ignore_label, 4: ignore_label, 5: ignore_label, 6: ignore_label,
                    7: 0, 8: 1, 9: ignore_label, 10: ignore_label, 11: 2, 12: 3, 13: 4,
                    14: ignore_label, 15: ignore_label, 16: ignore_label, 17: 5,
                    18: ignore_label, 19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 24: 11, 25: 12, 26: 13, 27: 14,
                    28: 15, 29: ignore_label, 30: ignore_label, 31: 16, 32: 17, 33: 18}

class City_Dataset(Dataset):
    def __init__(self, split='train', data_root=None, transform=None,mode='fine'):
        self.split = split
        self.mode=mode
        self.data_root=data_root
        self.nbr_classes=19
        self.data_list=self.__make_name_list(self.data_root,split)
        self.id_to_trainId=ID_TO_TRAINID
        self.palette=palette.CityScpates_palette
        self.transform = transform
        self.MEAN = [0.28689529, 0.32513294, 0.28389176]
        self.STD = [0.17613647, 0.18099176, 0.17772235]

    def __make_name_list(self,data_root,split):
        assert split in ['train','val'] and self.mode in ['fine','coarse','combined']
        SUFIX = '_gtFine_labelIds.png'
        image_paths, label_paths = [], []
        if self.mode == 'coarse':
            img_dir_name = 'leftImg8bit_trainextra' if self.split == 'train_extra' else 'leftImg8bit_trainvaltest'
            label_path = os.path.join(self.data_root, 'gtCoarse', 'gtCoarse', self.split)
        else:
            img_dir_name = 'leftImg8bit_trainvaltest'
            label_path = os.path.join(self.data_root, 'gtFine_trainvaltest', 'gtFine', self.split)
        image_path = os.path.join(self.data_root, img_dir_name, 'leftImg8bit', self.split)
        assert os.listdir(image_path) == os.listdir(label_path)

        for city in os.listdir(image_path):
            image_paths.extend(sorted(glob(os.path.join(image_path, city, '*.png'))))
            label_paths.extend(sorted(glob(os.path.join(label_path, city, f'*{SUFIX}'))))

        return list(zip(image_paths,label_paths))

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        input_path,label_path=self.data_list[index]
        input=Image.open(input_path)
        label=Image.open(label_path)

        if self.transform is not None:
            sample=self.transform({'img':input,'label':label})
        input,label=sample['img'],sample['label']
        for k, v in self.id_to_trainId.items():
            label[label == k] = v
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
                             transform.Crop(base_size=1024,crop_height=513,crop_width=513,type='central'),
                             transform.GaussianBlur(),
                             transform.ToTensor(),
                             transform.Normalize(std = [0.17613647, 0.18099176, 0.17772235],mean = [0.28689529, 0.32513294, 0.28389176])
                             ])

    split='train'
    data_root="D:\\datasets\\cityscapes\\"

    Voc_set=City_Dataset(split=split,data_root=data_root,transform=trans)
    col_map=Voc_set.palette
    dataloader = DataLoader(Voc_set, batch_size=10, shuffle=True, num_workers=0)

    val_list=[]
    for ii, sample in enumerate(dataloader):
        gt = sample[1][0]
        if (len(torch.unique(gt))>3):
            if len(val_list)<15:
                #print(gt.shape)
                gt=colorization(gt.numpy(),col_map)

                img=sample[0][0]
                img=transform.DeNormalize(std = [0.17613647, 0.18099176, 0.17772235],mean = [0.28689529, 0.32513294, 0.28389176])(img)
                #print(np.array(gt))
                gt=transforms.ToTensor()(gt.convert('RGB'))
                #print(gt)
                val_list.extend([img,gt])
            else:
                break
    val_list=torch.stack(val_list,0)
    val_list=make_grid(val_list,nrow=2,padding=3)
    val_list=transforms.ToPILImage()(val_list)
    val_list.show()










