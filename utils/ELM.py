import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
from scipy.ndimage.interpolation import rotate
import numpy as np


df_name=['up','down','left','right']

class Base_class(nn.Module):
    def __init__(self, nbr_classes=21,kernel_size=7, type='A',ignore_index=255):
        super().__init__()
        self.kernel_size = kernel_size
        self.ignore_index = ignore_index
        self.nbr_classes=nbr_classes
        self.smooth=1.
        self.half_kernel=kernel_size//2
        self.nbr_splitter=8 if type=='C' else 4
        self.type=type
        self.df_vector=self.anisot_initialization(self.type)

    def network_builder(self,indices):
        df_vector=self.df_vector[indices]
        #self.nbr_splitter=len(df_vector)
        weight = torch.zeros(self.nbr_classes, self.nbr_classes, self.kernel_size,
                              self.kernel_size).cuda().float()
        #print(df_vector[i])
        for k in range(self.nbr_classes):
            weight[k, k]=torch.from_numpy(df_vector.copy()).cuda().float()
        kernel_convolution=nn.Conv2d(self.nbr_classes,self.nbr_classes,kernel_size=self.kernel_size,stride=1,dilation=1, groups=1,
                  bias=False)
        kernel_convolution.weight=nn.Parameter(weight)
        for param in kernel_convolution.parameters():
            param.requires_grad = False
        return kernel_convolution


    def anisot_initialization(self,type='A'):
        aniso_vector=[]
        base_matrix=np.zeros((self.kernel_size,self.kernel_size))
        mid = int(self.kernel_size / 2)+1

        if type =='C':
            matrix_A=self.anisot_initialization(type='A')
            matrix_B=self.anisot_initialization(type='B')
            for i in range(self.nbr_splitter):
                aniso_vector.append(matrix_A[i])
                aniso_vector.append(matrix_B[i])
        else:
            if type == 'A':
                start = [x for x in range(mid)]
                base_matrix[mid - 1, start] = 1
            elif type == 'B':
                matrix = np.ones(mid)
                matrix = np.diag(matrix)
                base_matrix[np.where(matrix)] = 1
            for i in range(self.nbr_splitter):
                base_matrix=np.rot90(base_matrix,i)
                aniso_vector.append(base_matrix)
        return aniso_vector




class Point_loss(Base_class):
    def forward(self, prediction, class_label):
        point_loss=0
        prediction = F.softmax(prediction, dim=1)
        batch,nbr_class,w,h=prediction.size()
        for i in range(self.nbr_splitter):
            aniso_opator = self.network_builder(i)
            tmp_pre=aniso_opator(F.pad(prediction, pad=(self.kernel_size // 2, self.kernel_size // 2, self.kernel_size // 2, self.kernel_size // 2), mode='replicate')).view(batch,nbr_class,w,h)
            tmp_tar=aniso_opator(F.pad(class_label, pad=(self.kernel_size // 2, self.kernel_size // 2, self.kernel_size // 2, self.kernel_size // 2), mode='replicate')).view(batch,nbr_class,w,h)
            point_loss+=nn.MSELoss()(tmp_tar,tmp_pre)

        return point_loss/self.nbr_splitter

class Equipotential_line_loss(Base_class):
    def forward(self, prediction, class_label,mu=10):
        prediction = F.softmax(prediction, dim=1)
        batch, nbr_class, w, h = prediction.size()
        line_loss = 0
        for i in range(int(self.nbr_splitter)):
            aniso_opator = self.network_builder(i)
            tmp_pre = aniso_opator(F.pad(prediction, pad=(
                self.half_kernel, self.half_kernel, self.half_kernel, self.half_kernel),
                                         mode='constant')).view(batch * nbr_class, w * h)
            tmp_tar = aniso_opator(F.pad(class_label, pad=(
                self.half_kernel, self.half_kernel, self.half_kernel, self.half_kernel),
                                         mode='constant')).view(batch * nbr_class, w * h)

            pre_indices = torch.argsort(tmp_pre, dim=-1)
            label_mask = (tmp_tar >= 1) & (tmp_tar <= self.half_kernel)
            valid_dims = torch.unique(torch.stack(torch.where(label_mask), dim=1)[:, 0])
            test_mask = torch.zeros(batch * nbr_class, w * h).cuda()
            # print(i)
            for dim in valid_dims:
                start = (tmp_tar[dim] < 1).sum()
                end = (tmp_tar[dim] < (self.half_kernel + 1)).sum()
                test_mask[dim][pre_indices[dim][start:end]] = 1

            tmp_pre = (tmp_pre * test_mask).contiguous()
            tmp_tar = (tmp_tar * label_mask).contiguous()
            for k in range(1, self.half_kernel + 1):
                p_t_k = torch.exp(-((tmp_pre - k) ** mu))
                p_g_k = torch.exp(-((tmp_tar - k) ** mu))
                intersection = (p_t_k * p_g_k).sum()
                loss = 1 - ((2. * (intersection + self.smooth) * p_g_k.sum()) / (
                        (p_g_k.sum() + p_t_k.sum() + self.smooth) * ((p_g_k * p_g_k).sum())))
                line_loss += loss

        return line_loss / self.nbr_splitter


class Equipotential_learning(nn.Module):
    def __init__(self, nbr_classes=21,point=['A',7],line=['A',7],mu=10,ignore_index=255):
        super().__init__()
        type,kernel=point
        self.point_loss=Point_loss(nbr_classes=nbr_classes,kernel_size=kernel,type=type)
        self.nbr_classes=nbr_classes
        type, kernel = line
        self.line_loss=Equipotential_line_loss(nbr_classes=nbr_classes,kernel_size=kernel,type=type)
        self.mu=mu
    def forward(self, prediction, class_label):
        point_loss=self.point_loss(prediction,class_label)
        line_loss=self.line_loss(prediction,class_label,self.mu)

        return 0.02*point_loss+0.2*line_loss



if __name__=='__main__':
    input=torch.rand(1,21,512,512).cuda().float()
    one_hot=(torch.rand(1,21,512,512)>0.5).cuda().float()
    loss=Equipotential_learning(nbr_classes=21,point=['A',7],line=['A',7],mu=10)
    print(loss(input,one_hot))





