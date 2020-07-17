from base import BaseModel
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from itertools import chain
from torchsummary import summary
from torchvision import models
import torch.utils.model_zoo as model_zoo
from utils.helpers import initialize_weights
import os
from torch.utils import tensorboard
from utils.transforms import Predinormalize
from torchvision.models import vgg19
from PIL import Image,ImageFilter
import snowy
import matplotlib.pyplot as plt
import cv2
import time
## deeplab part
class ResNet(nn.Module):
    def __init__(self, in_channels=3, output_stride=16, backbone='resnet101', pretrained=True):
        super(ResNet, self).__init__()
        model = getattr(models, backbone)(pretrained)
        if not pretrained or in_channels != 3:
            self.layer0 = nn.Sequential(
                nn.Conv2d(in_channels, 64, 7, stride=2, padding=3, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            )
            initialize_weights(self.layer0)
        else:
            self.layer0 = nn.Sequential(*list(model.children())[:4])

        self.layer1 = model.layer1
        self.layer2 = model.layer2
        self.layer3 = model.layer3
        self.layer4 = model.layer4

        if output_stride == 16:
            s3, s4, d3, d4 = (2, 1, 1, 2)
        elif output_stride == 8:
            s3, s4, d3, d4 = (1, 1, 2, 4)

        if output_stride == 8:
            for n, m in self.layer3.named_modules():
                if 'conv1' in n and (backbone == 'resnet34' or backbone == 'resnet18'):
                    m.dilation, m.padding, m.stride = (d3, d3), (d3, d3), (s3, s3)
                elif 'conv2' in n:
                    m.dilation, m.padding, m.stride = (d3, d3), (d3, d3), (s3, s3)
                elif 'downsample.0' in n:
                    m.stride = (s3, s3)

        for n, m in self.layer4.named_modules():
            if 'conv1' in n and (backbone == 'resnet34' or backbone == 'resnet18'):
                m.dilation, m.padding, m.stride = (d4, d4), (d4, d4), (s4, s4)
            elif 'conv2' in n:
                m.dilation, m.padding, m.stride = (d4, d4), (d4, d4), (s4, s4)
            elif 'downsample.0' in n:
                m.stride = (s4, s4)

    def forward(self, x):
        x = self.layer0(x)
        x = self.layer1(x)
        low_level_features = x
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x, low_level_features


''' 
-> (Aligned) Xception BackBone
Pretrained model from https://github.com/Cadene/pretrained-models.pytorch
by Remi Cadene
'''


class SeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1, bias=False,
                 BatchNorm=nn.BatchNorm2d):
        super(SeparableConv2d, self).__init__()

        if dilation > kernel_size // 2:
            padding = dilation
        else:
            padding = kernel_size // 2

        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding=padding,
                               dilation=dilation, groups=in_channels, bias=bias)
        self.bn = nn.BatchNorm2d(in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, 1, bias=bias)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn(x)
        x = self.pointwise(x)
        return x


class Block(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, dilation=1, exit_flow=False, use_1st_relu=True):
        super(Block, self).__init__()

        if in_channels != out_channels or stride != 1:
            self.skip = nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False)
            self.skipbn = nn.BatchNorm2d(out_channels)
        else:
            self.skip = None

        rep = []
        self.relu = nn.ReLU(inplace=True)

        rep.append(self.relu)
        rep.append(SeparableConv2d(in_channels, out_channels, 3, stride=1, dilation=dilation))
        rep.append(nn.BatchNorm2d(out_channels))

        rep.append(self.relu)
        rep.append(SeparableConv2d(out_channels, out_channels, 3, stride=1, dilation=dilation))
        rep.append(nn.BatchNorm2d(out_channels))

        rep.append(self.relu)
        rep.append(SeparableConv2d(out_channels, out_channels, 3, stride=stride, dilation=dilation))
        rep.append(nn.BatchNorm2d(out_channels))

        if exit_flow:
            rep[3:6] = rep[:3]
            rep[:3] = [
                self.relu,
                SeparableConv2d(in_channels, in_channels, 3, 1, dilation),
                nn.BatchNorm2d(in_channels)]

        if not use_1st_relu: rep = rep[1:]
        self.rep = nn.Sequential(*rep)

    def forward(self, x):
        output = self.rep(x)
        if self.skip is not None:
            skip = self.skip(x)
            skip = self.skipbn(skip)
        else:
            skip = x

        x = output + skip
        return x


class Xception(nn.Module):
    def __init__(self, output_stride=16, in_channels=3, pretrained=True):
        super(Xception, self).__init__()

        # Stride for block 3 (entry flow), and the dilation rates for middle flow and exit flow
        if output_stride == 16: b3_s, mf_d, ef_d = 2, 1, (1, 2)
        if output_stride == 8: b3_s, mf_d, ef_d = 1, 2, (2, 4)

        # Entry Flow
        self.conv1 = nn.Conv2d(in_channels, 32, 3, 2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(32, 64, 3, 1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(64)

        self.block1 = Block(64, 128, stride=2, dilation=1, use_1st_relu=False)
        self.block2 = Block(128, 256, stride=2, dilation=1)
        self.block3 = Block(256, 728, stride=b3_s, dilation=1)

        # Middle Flow
        for i in range(16):
            exec(f'self.block{i + 4} = Block(728, 728, stride=1, dilation=mf_d)')

        # Exit flow
        self.block20 = Block(728, 1024, stride=1, dilation=ef_d[0], exit_flow=True)

        self.conv3 = SeparableConv2d(1024, 1536, 3, stride=1, dilation=ef_d[1])
        self.bn3 = nn.BatchNorm2d(1536)
        self.conv4 = SeparableConv2d(1536, 1536, 3, stride=1, dilation=ef_d[1])
        self.bn4 = nn.BatchNorm2d(1536)
        self.conv5 = SeparableConv2d(1536, 2048, 3, stride=1, dilation=ef_d[1])
        self.bn5 = nn.BatchNorm2d(2048)

        initialize_weights(self)
        if pretrained: self._load_pretrained_model()

    def _load_pretrained_model(self):
        url = 'http://data.lip6.fr/cadene/pretrainedmodels/xception-b5690688.pth'
        pretrained_weights = model_zoo.load_url(url)
        state_dict = self.state_dict()
        model_dict = {}

        for k, v in pretrained_weights.items():
            if k in state_dict:
                if 'pointwise' in k:
                    v = v.unsqueeze(-1).unsqueeze(-1)  # [C, C] -> [C, C, 1, 1]
                if k.startswith('block11'):
                    # In Xception there is only 8 blocks in Middle flow
                    model_dict[k] = v
                    for i in range(8):
                        model_dict[k.replace('block11', f'block{i + 12}')] = v
                elif k.startswith('block12'):
                    model_dict[k.replace('block12', 'block20')] = v
                elif k.startswith('bn3'):
                    model_dict[k] = v
                    model_dict[k.replace('bn3', 'bn4')] = v
                elif k.startswith('conv4'):
                    model_dict[k.replace('conv4', 'conv5')] = v
                elif k.startswith('bn4'):
                    model_dict[k.replace('bn4', 'bn5')] = v
                else:
                    model_dict[k] = v

        state_dict.update(model_dict)
        self.load_state_dict(state_dict)

    def forward(self, x):
        # Entry flow
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.block1(x)
        low_level_features = x
        x = F.relu(x)
        x = self.block2(x)
        x = self.block3(x)

        # Middle flow
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.block7(x)
        x = self.block8(x)
        x = self.block9(x)
        x = self.block10(x)
        x = self.block11(x)
        x = self.block12(x)
        x = self.block13(x)
        x = self.block14(x)
        x = self.block15(x)
        x = self.block16(x)
        x = self.block17(x)
        x = self.block18(x)
        x = self.block19(x)

        # Exit flow
        x = self.block20(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)

        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu(x)

        x = self.conv5(x)
        x = self.bn5(x)
        x = self.relu(x)

        return x, low_level_features


''' 
-> The Atrous Spatial Pyramid Pooling
'''


def assp_branch(in_channels, out_channles, kernel_size, dilation):
    padding = 0 if kernel_size == 1 else dilation
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channles, kernel_size, padding=padding, dilation=dilation, bias=False),
        nn.BatchNorm2d(out_channles),
        nn.ReLU(inplace=True))


class ASSP(nn.Module):
    def __init__(self, in_channels, output_stride):
        super(ASSP, self).__init__()

        assert output_stride in [8, 16], 'Only output strides of 8 or 16 are suported'
        if output_stride == 16:
            dilations = [1, 6, 12, 18]
        elif output_stride == 8:
            dilations = [1, 12, 24, 36]

        self.aspp1 = assp_branch(in_channels, 256, 1, dilation=dilations[0])
        self.aspp2 = assp_branch(in_channels, 256, 3, dilation=dilations[1])
        self.aspp3 = assp_branch(in_channels, 256, 3, dilation=dilations[2])
        self.aspp4 = assp_branch(in_channels, 256, 3, dilation=dilations[3])

        self.avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(in_channels, 256, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True))

        self.conv1 = nn.Conv2d(256 * 5, 256, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(256)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(0.5)

        initialize_weights(self)

    def forward(self, x):
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x5 = F.interpolate(self.avg_pool(x), size=(x.size(2), x.size(3)), mode='bilinear', align_corners=True)

        x = self.conv1(torch.cat((x1, x2, x3, x4, x5), dim=1))
        x = self.bn1(x)
        x = self.dropout(self.relu(x))

        return x
class Decoder(nn.Module):
    def __init__(self, low_level_channels, num_classes):
        super(Decoder, self).__init__()
        self.conv1 = nn.Conv2d(low_level_channels, 48, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(48)
        self.relu = nn.ReLU(inplace=True)

        # Table 2, best performance with two 3x3 convs
        self.output = nn.Sequential(
            nn.Conv2d(48 + 256, 256, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Conv2d(256, num_classes, 1, stride=1),
        )
        initialize_weights(self)

    def forward(self, x, low_level_features):
        low_level_features = self.conv1(low_level_features)
        low_level_features = self.relu(self.bn1(low_level_features))
        H, W = low_level_features.size(2), low_level_features.size(3)

        x = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=True)
        x = self.output(torch.cat((low_level_features, x), dim=1))
        return x
## Networks above belong to Depplab_plus

class kernel_block(nn.Module):
    def __init__(self,in_channels,out_channels,avg=True,kernel=3,stride=1):
        super(kernel_block, self).__init__()
        self.conv=nn.Sequential(
            nn.Conv2d(in_channels,out_channels,kernel_size=kernel,stride=stride,padding=1),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=kernel, stride=stride, padding=1),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=kernel, stride=stride, padding=1),
            nn.ReLU()
        )
        self.avg=avg
    def forward(self,x):
        if self.avg:
            x=nn.AvgPool2d((2,2))(x)
        x=self.conv(x)
        return x




class kernel_computation(nn.Module):
    def __init__(self,kernel_size=5):
        super(kernel_computation, self).__init__()
        self.kernel_size=kernel_size
        self.unfolder=nn.Unfold(kernel_size=kernel_size,dilation=1,padding=kernel_size//2,stride=1)

    def forward(self,coarse_prediction,kernel_collection):
        #start=time.time()
        kernel=self.kernel_size
        batch,channel,height,width=kernel_collection.size()
        kernel_collection=kernel_collection.permute(0,2,3,1)
        #print(kernel_collection.shape)
        weights = kernel_collection.view(-1, height * width, kernel*kernel)  ##batch*(w*h)*kernel*kernel
        #Pad_coarse_prediction=F.pad(coarse_prediction,[kernel//2,kernel//2,kernel//2,kernel//2])

        #print(coarse_prediction.shape)
        ##coarse_prediction=coarse_prediction.view(coarse_prediction.shape[0],1,coarse_prediction.shape[1],coarse_prediction.shape[2])
        unfolded=self.unfolder(coarse_prediction)##b*(c*kernel*kernel)*(w*h)

        unfolded=unfolded.view(coarse_prediction.shape[0],coarse_prediction.shape[1],kernel*kernel,-1)
        unfolded=unfolded.permute(0,1,3,2)
        #print(unfolded.shape)
        weights=torch.stack([weights]*coarse_prediction.shape[1],dim=1)
        result=torch.sum(unfolded.mul(weights),dim=-1)
        result=result.view(result.shape[0],result.shape[1],height,width)

        return result

class DeNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        return tensor

class kernel_prediction(nn.Module):
    def __init__(self,nbr_classes,out_channels,up_method):
        super(kernel_prediction, self).__init__()
        self.up_method=up_method
        self.conv1 = kernel_block(3,64,avg=False)
        self.conv2 = kernel_block(128, 128, avg=True)
        self.conv3 = kernel_block(256,256, avg=True)
        self.conv4 = kernel_block(512, 512, avg=True)
        self.conv5 = kernel_block(1024, 1024, avg=True)


        self.pro_conv1 = kernel_block(nbr_classes,64,avg=False)
        self.pro_conv2 = kernel_block(64, 128, avg=True)
        self.pro_conv3 = kernel_block(128, 256, avg=True)
        self.pro_conv4 = kernel_block(256,512, avg=True)
        self.pro_conv5 = kernel_block(512, 1024, avg=True)

        self.conv6 = kernel_block(2048+1024, 1024, avg=False)
        self.conv7 = kernel_block(512+1024, 512, avg=False)
        self.conv8 = kernel_block(256+512, 256, avg=False)
        self.conv9 =kernel_block(128+256,out_channels,avg=False)
        self.out=nn.Conv2d(out_channels,out_channels,1,1,0)

    def forward(self,x,pro):
        conv1=self.conv1(x)
        pro_conv1=self.pro_conv1(pro)
        fusion_1=torch.cat([conv1,pro_conv1],dim=1)## [N*128*256*256]

        conv2 = self.conv2(fusion_1)
        pro_conv2=self.pro_conv2(pro_conv1)
        fusion_2=torch.cat([conv2,pro_conv2],dim=1)##[N*256*256*256]

        conv3 = self.conv3(fusion_2)
        pro_conv3=self.pro_conv3(pro_conv2)
        fusion_3 =torch.cat([conv3,pro_conv3],dim=1) ## [N*512*256*256]

        conv4 = self.conv4(fusion_3)
        pro_conv4=self.pro_conv4(pro_conv3)
        fusion_4=torch.cat([conv4,pro_conv4],dim=1)##[N*1024]

        conv5 = self.conv5(fusion_4)
        pro_conv5=self.pro_conv5(pro_conv4)
        fusion_5=torch.cat([conv5,pro_conv5],dim=1)##[N*2048]
        fusion_5=nn.Dropout(0.5)(fusion_5)
        x=torch.cat([fusion_4,F.interpolate(fusion_5,scale_factor=2,mode=self.up_method,align_corners=False)],dim=1)
        conv6 = self.conv6(x)
        x = torch.cat([fusion_3, F.interpolate(conv6, scale_factor=2, mode=self.up_method,align_corners=False)],dim=1)
        conv7 = self.conv7(x)

        x = torch.cat([fusion_2, F.interpolate(conv7, scale_factor=2, mode=self.up_method,align_corners=False)],dim=1)
        conv8=self.conv8(x)
        x = torch.cat([fusion_1, F.interpolate(conv8, scale_factor=2, mode=self.up_method,align_corners=False)],dim=1)
        x=self.conv9(x)
        output=self.out(x)
        return output


class DeepLab(nn.Module):
    def __init__(self, num_classes, in_channels=3, backbone='xception', pretrained=True,
                 output_stride=16, freeze_bn=False, **_):

        super(DeepLab, self).__init__()
        assert ('xception' or 'resnet' in backbone)
        if 'resnet' in backbone:
            self.backbone = ResNet(in_channels=in_channels, output_stride=output_stride, pretrained=pretrained)
            low_level_channels = 256
        else:
            self.backbone = Xception(output_stride=output_stride, pretrained=pretrained)
            low_level_channels = 128

        self.ASSP = ASSP(in_channels=2048, output_stride=output_stride)
        self.decoder = Decoder(low_level_channels, num_classes)

        if freeze_bn: self.freeze_bn()

    def forward(self, x):
        H, W = x.size(2), x.size(3)
        x, low_level_features = self.backbone(x)
        x = self.ASSP(x)
        x = self.decoder(x, low_level_features)
        x = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=True)
        return x

class Kernel_net(BaseModel):
    def __init__(self, num_classes, in_channels=3, kerne_size=5,backbone='resnet101', freeze_bn=False, **_):

        super(Kernel_net, self).__init__()
        self.back_arch=DeepLab(num_classes, in_channels=in_channels, backbone=backbone, pretrained=True)
        self.kernel_network=kernel_prediction(num_classes,kerne_size*kerne_size,up_method='bilinear')
        self.predition_transform=Predinormalize()
        self.conv_computation=kernel_computation()
        self.distance_loss=nn.L1Loss()
        if freeze_bn: self.freeze_bn()

    def test_kernel(self,input,input2):
        kernel_output = self.kernel_network(input)
        coarse_pred=self.back_arch(input2)
        coarse_pro=F.softmax(coarse_pred,dim=1)
        coarse_pro,__=torch.max(coarse_pro,dim=1)
        coarse_pro=self.predition_transform(coarse_pro)
        out=self.conv_computation(coarse_pro,kernel_output)
        return out

    def Distance_extraction(self,prediction, target):
        prediction_stack=torch.clamp(prediction,-5,5)
        target=torch.clamp(target,-5,5)
        loss = self.distance_loss(prediction_stack, target)
        return loss

    def forward(self, x,distance):
        coarse_pre=self.back_arch(x) ##B*N*W*H
        pro_field=F.softmax(coarse_pre,dim=1)##B*N*W*H
        #coarse_pro,__=torch.max(pro_field,dim=1)## probability field
        coarse_estimation=self.predition_transform(pro_field)
        kernel_output = self.kernel_network(x,pro_field)
        out=self.conv_computation(coarse_estimation,kernel_output)
        distance_loss=self.Distance_extraction(out,distance)
        return coarse_pre,distance_loss

    def kernel_visualization(self,x):
        coarse_pre=self.back_arch(x) ##B*N*W*H
        pro_field=F.softmax(coarse_pre,dim=1)##B*N*W*H
        ##coarse_pro,__=torch.max(pro_field,dim=1)## probability field
        coarse_estimation=self.predition_transform(pro_field)
        kernel_output = self.kernel_network(x,pro_field)
        out=self.conv_computation(coarse_estimation,kernel_output)


        return kernel_output,coarse_estimation,out

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_normal_(m.weight.data)
        elif isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight.data)

    def get_backbone_params(self):
        return self.back_arch.backbone.parameters()

    def get_decoder_params(self):
        return chain(self.back_arch.ASSP.parameters(), self.back_arch.decoder.parameters())

    def get_kernel_params(self):
        return self.kernel_network.parameters()

    def freeze_bn(self):
        for module in self.modules():
            if isinstance(module, nn.BatchNorm2d): module.eval()


# #
# if __name__=='__main__':
#     path='../distance_transform/'
#     file=os.listdir(path)
#     kernel_set=[]
#     image_set=[]
#     name=['2007_000175','2007_000323','2007_000346','2007_000999','2007_001185']
#     for img in name:
#         #print(path+'/input/'+str(img)+'.jpg')
#         image_set.append(torch.from_numpy(np.transpose(cv2.resize(cv2.imread(path+'input/'+str(img)+'.jpg',3),(256,256)),(2,0,1))))
#         #print(''+path+str(name)+'.png')
#         kernel_set.append(torch.from_numpy(np.transpose(cv2.resize(cv2.imread(''+path+str(img)+'.png',3),(256,256)),(2,0,1))))
#     image_set = torch.stack(image_set, dim=0).float().cuda()
#     kernel_set = torch.stack(kernel_set, dim=0).float().cuda()
#     model=Kernel_net(num_classes=21,in_channels=3,kerne_size=5).to('cuda')
#     ## resume
#
#     checkpoint = torch.load('../best_model.pth')
#     from collections import OrderedDict
#
#     new_state_dict = OrderedDict()
#     for k, v in checkpoint['state_dict'].items():
#         name = k[7:]  # remove 'module.' of dataparallel
#         new_state_dict[name] = v
#
#     model.back_arch.load_state_dict(new_state_dict)
#
#     output=model.test_kernel(image_set,kernel_set)
#     for idx,img in enumerate(output):
#         x=(img[0]-img[0].min())/(img[0].max()-img[0].min())
#         x=x.view(256,256,1)
#         cv2.imwrite(''+str(idx)+'.png',x.data.cpu().numpy()*255)


# if __name__=='__main__':
#     model=kernel_prediction(out_channels=25,up_method='bilinear').cuda()
#     rgb_input=(3,256,256)
#     pro_input=(1,256,256)
#     writer=tensorboard.SummaryWriter('./graph')
#     writer.add_graph(model,[torch.rand(5,3,256,256).cuda(),torch.rand(5,1,256,256).cuda()])
#     writer.close()