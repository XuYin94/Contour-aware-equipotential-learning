# Originally written by Kazuto Nakashima
# https://github.com/kazuto1011/deeplab-pytorch

from base import BaseDataSet, BaseDataLoader
from utils import palette
import numpy as np
import os
import scipy
import torch
from PIL import Image
from utils.Signed_distance_field import Distance_transfer, sdf_generator
import cv2
from torch.utils.data import Dataset
from torchvision import transforms


class TESTDataset(BaseDataSet):
    """
    Pascal Voc dataset
    http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
    """

    def __init__(self, **kwargs):
        self.num_classes = 2
        self.palette = palette.Test_palette
        super(TESTDataset, self).__init__(**kwargs)

    def _set_files(self):
        self.image_dir = os.path.join(self.root, 'train')
        self.label_dir = os.path.join(self.root, 'train_masks')
        self.files = os.listdir(self.image_dir)

    def _load_data(self, index):
        image_id = self.files[index][:-4]
        image_path = os.path.join(self.image_dir, image_id + '.jpg')
        label_path = os.path.join(self.label_dir, image_id + '_mask.png')
        image = np.asarray(Image.open(image_path).resize((256, 256)), dtype=np.float32)
        label = np.asarray(Image.open(label_path).resize((256, 256)), dtype=np.int32)
        distance = np.asarray(Distance_transfer(label_path), dtype=np.float32)
        image_id = self.files[index].split("/")[-1].split(".")[0]
        return image, label, distance, image_id


class TESTAugDataset(BaseDataSet):
    """
    Contrains both SBD and VOC 2012 dataset
    Annotations : https://github.com/DrSleep/tensorflow-deeplab-resnet#evaluation
    Image Sets: https://ucla.app.box.com/s/rd9z2xvwsfpksi7mi08i2xqrj7ab4keb/file/55053033642
    """

    def __init__(self, **kwargs):
        self.num_classes = 21
        self.palette = palette.get_voc_palette(self.num_classes)
        super(TESTAugDataset, self).__init__(**kwargs)

    def _set_files(self):
        #self.root = os.path.join(self.root, 'VOCdevkit/VOC2012')

        self.image_dir = os.path.join(self.root, 'aug')
        self.label_dir = os.path.join(self.root, 'aug_mask')
        self.files = os.listdir(self.image_dir)

    def _load_data(self, index):
        image_id = self.files[index][:-4]
        image_path = os.path.join(self.image_dir, image_id + '.jpg')
        label_path = os.path.join(self.label_dir, image_id + '_mask.png')
        #print(label_path)
        image = np.asarray(Image.open(image_path).resize((256, 256)), dtype=np.float32)
        label = np.asarray(Image.open(label_path).resize((256, 256)), dtype=np.int32)
        distance = np.asarray(Distance_transfer(label_path), dtype=np.float32)
        image_id = self.files[index].split("/")[-1].split(".")[0]
        return image, label, distance, image_id


class TEST(BaseDataLoader):
    def __init__(self, data_dir, batch_size, split, crop_size=None, base_size=None, scale=True, num_workers=1,
                 val=False,
                 shuffle=False, flip=False, rotate=False, blur=False, augment=False, val_split=None, return_id=False):

        self.MEAN = [0.45734706, 0.43338275, 0.40058118]
        self.STD = [0.23965294, 0.23532275, 0.2398498]

        kwargs = {
            'root': data_dir,
            'split': split,
            'mean': self.MEAN,
            'std': self.STD,
            'augment': augment,
            'crop_size': crop_size,
            'base_size': base_size,
            'scale': scale,
            'flip': flip,
            'blur': blur,
            'rotate': rotate,
            'return_id': return_id,
            'val': val
        }
        #print(split)
        if split in ["val"]:
            self.dataset = TESTAugDataset(**kwargs)
        elif split in ["train",  "test"]:
            self.dataset = TESTDataset(**kwargs)
        else: raise ValueError(f"Invalid split name {split}")
        #self.dataset = TESTDataset(**kwargs)
        super(TEST, self).__init__(self.dataset, batch_size, shuffle, num_workers, val_split)

