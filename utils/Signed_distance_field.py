
import PIL
from PIL import Image, ImageFilter
import cv2
import numpy as np
import snowy
import os
from matplotlib import pyplot as plt
import torch
import  torch.nn as nn
import torch.nn.functional as F
import numpy as np
INF = 1e20
# img = Image.open('../distance_transform/ADE_train_00000001.png')
# img = img.convert("L")
# img=np.expand_dims(np.array(img),axis=0)
# img=np.expand_dims(np.array(img),axis=0)
# #Calculating Edges using the passed laplican Kernel
# #final_2 = img.filter(ImageFilter.Kernel((3, 3), (-1, -1, -1, -1, 8,-1, -1, -1, -1), 1, 0))
# conv=nn.Conv2d(in_channels=1,out_channels=1, kernel_size=3, stride=1, padding=1)
# filter=np.array((-1, -1, -1, -1, 8,-1, -1, -1, -1)).reshape(3,3)
#
# #print(conv.weight.shape)
# #print(np.expand_dims(np.expand_dims(filter,axis=0),axis=0).shape)
# conv.weight.data=torch.from_numpy(np.expand_dims(np.expand_dims(filter,axis=0),axis=0))
# #print(torch.from_numpy(img).shape)
# img=conv(torch.from_numpy(img).long())
# img=img.cpu().data.numpy()[0][0]
# img=img!=0
# img=np.expand_dims(img,axis=-1)
# sdf = snowy.unitize(snowy.generate_sdf(img))
# #print(sdf[sdf==0])
# snowy.show(sdf)
#final = np.array(final_2)

# final.save('new_name.png')
# edges = cv2.Canny(img,0.5,1)
# edges=edges!=0
# final = final != 0
# edges = np.expand_dims(final, axis=-1)
# # print(edges.shape)
# sdf = snowy.unitize(snowy.generate_sdf(edges))
# snowy.show(snowy.hstack([edges, sdf]))

def Example_transfer(path,nbr_classes,idx):
    #print(path)
   # print(Image.open(path).copy())
    img = np.asarray(Image.open(path).resize((128,128))).copy()
    if(255 in img):
        img=img.copy()
        img.setflags(write=1)
        img[img==255]=0
    value=1000
    #distance_helper=np.full((img.shape[0],img.shape[1]),value)
    distance_label=np.full((nbr_classes,img.shape[0],img.shape[1]),value)
    channel_index=np.unique(img)
    label=img!=0
    edges=np.expand_dims(label,axis=-1)
    sdf=snowy.generate_sdf(edges)
    #snowy.show(snowy.unitize(sdf))
    before=sdf[:,:,0]
    cv2.imwrite('./'+str(idx)+'_before.png',255*(before-np.amin(before))/(np.amax(before)-np.amin(before)))
    for index in channel_index:
        distance_label[index]=np.where(img==index,sdf[:,:,0],value)

    test=np.full((img.shape[0],img.shape[1]),value)
    for index in channel_index:
        test=np.where(distance_label[index]!=value,distance_label[index],test)
    #test=
    cv2.imwrite('./' + str(idx) + '_after.png', 255*(test-np.amin(test))/(np.amax(test)-np.amin(test)))





def Distance_transfer(path,nbr_classes,size):
    #print(path)
    img = np.asarray(Image.open(path).resize((size,size))).copy()
    if(255 in img):
        img=img.copy()
        img.setflags(write=1)
        img[img==255]=0
    value=1000
    distance_label=np.full((nbr_classes,img.shape[0],img.shape[1]),value)
    channel_index=np.unique(img)
    label=img!=0
    edges=np.expand_dims(label,axis=-1)
    sdf=snowy.generate_sdf(edges)
    before=sdf[:,:,0]
    for index in channel_index:
        distance_label[index]=np.where(img==index,before,value)
    distance_label=np.transpose(distance_label,[1,2,0])
    return distance_label
#
#

def sdf_generator(path):
    img = cv2.imread(path)
    img=cv2.resize(img,(224,224))
    edges = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(edges, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)


    #print(edges)
    imag=cv2.distanceTransform(thresh,cv2.DIST_L2,cv2.DIST_MASK_3)
    result = cv2.normalize(imag, None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8UC1)
    cv2.imshow('test',result)
    cv2.waitKey(-1)
    # imag=imag>=0.9
    # imag=np.expand_dims(imag,axis=-1)
    # sdf = snowy.unitize(snowy.generate_sdf(imag))
    # snowy.show(sdf)



# if __name__=="__main__":
#     path='../distance_transform/'
#     #name=["2007_000032","2007_000039","2007_000063","2007_000068","2007_000121","2007_000170","2007_000241","2007_000243","2007_000250","2007_000256","2007_000333"]
#     for idx,name in enumerate(os.listdir(path)):
#         img_name=path+name
#         Example_transfer(img_name,21,idx)
        #signed_distance(path+name)

