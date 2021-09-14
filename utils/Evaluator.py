import numpy as np
import torch
import os
import glob
import datetime
from utils.utils import Check_dir
from torch.utils import tensorboard
import torch.nn.functional as F
from torchvision.utils import make_grid
from utils.utils import colorization
from torchvision import transforms
from utils import transform
import json

value_scale = 255
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

class Exp_Recorder(object):

    def __init__(self,configs,palette,mean=mean,std=std):
        self.configs=configs
        self.exp_time=datetime.datetime.now().strftime("%m_%d_%H_%M")
        self.saver_folder=os.path.join("run",self.configs["Data_loader"]["Dataset_name"],self.configs["Exp_type"],self.exp_time)
        Check_dir(self.saver_folder)
        self.writer=tensorboard.SummaryWriter(self.saver_folder)
        self.best_pred=0.0
        self.palette=palette

        self.restore_trans = transforms.Compose([transform.DeNormalize(mean=mean,std=std),
                                             transforms.ToPILImage()])

        self.viz_transform = transforms.Compose([transforms.Resize((512,512)),
                                              transforms.ToTensor()])

        self.potential_trans = transforms.Compose([transform.df_viz(),
                                                transforms.ToPILImage()])


    def _save_model(self,state,epoch,class_Iou):
        save_name=os.path.join(self.saver_folder,'model_'+str(epoch)+'.pth')
        if(epoch>=1):
            delete_name=self.saver_folder+'/model_'+str(epoch-1)+'.pth'
            if(os.path.exists(delete_name)):
                os.remove(delete_name)
        torch.save(state,save_name)
        pre=state['best_pre']
        if self.best_pred<pre:
            self.best_pred=pre
            with open(os.path.join(self.saver_folder,'best_pred.txt'),'w') as f:
                f.write("Mean Iou:"+str(self.best_pred)+"\n")
                for i in range(len(class_Iou)):
                    f.write("Class_"+str(i)+" result: "+str(class_Iou[i])+"\n")
            best_name=os.path.join(self.saver_folder,'best_model.pth')
            torch.save(state,best_name)


    def _save_exp_result(self):
        save_name=os.path.join(self.saver_folder,'Exp_config.json')
        with open(save_name,'w') as handle:
            json.dump(self.configs,handle, indent=4, sort_keys=True)

    def _update_writer(self,seg_loss,dis_loss,loss,seg_metric,optimizer,step,mode='train'):
        self.writer.add_scalar(''+str(mode)+'/dis_loss',dis_loss.item(),step)
        self.writer.add_scalar(''+ str(mode) + '/seg_loss', seg_loss.item(), step)
        self.writer.add_scalar(''+str(mode)+'/total_loss',loss.item(),step)

        for key, value in list(seg_metric.items())[:-1]:
            self.writer.add_scalar(f'{mode}/{key}',value,step)
        for i , para_group in enumerate(optimizer.param_groups):
            self.writer.add_scalar(f'{mode}/{str(i)}',para_group['lr'],step)

    def _update_visual(self,val_list, epoch):
        seg_val=[]
        pot_val=[]
        for idx, (i,o,t,pt_pre,pt_tar) in enumerate(val_list):
            valid_class=np.unique(t)[:3]  ## get the valid dimension of class index
            i = self.restore_trans(i)
            o,t = colorization(o,self.palette),colorization(t,self.palette)
            i,o,t = i.convert('RGB'),o.convert('RGB'),t.convert('RGB')
            [i,o,t] = [self.viz_transform(x) for x in [i,o,t]]
            seg_val.extend([i,o,t])

            pot_val.extend([i,o])
            for c_index in valid_class:## we just visualize the in the "up" direction
                valid_tar,valid_pre=pt_tar[0][c_index],pt_pre[0][c_index]
                valid_tar,valid_pre=self.potential_trans(valid_tar),self.potential_trans(valid_pre)
                valid_tar, valid_pre=valid_tar.convert('RGB'),valid_pre.convert('RGB')
                valid_tar, valid_pre=self.viz_transform(valid_tar),self.viz_transform(valid_pre)
                pot_val.extend([valid_tar,valid_pre])


        seg_val=torch.stack(seg_val,0)
        seg_val=make_grid(seg_val,nrow=3,padding=5)
        self.writer.add_image('seg_val',seg_val,epoch)

        pot_val=torch.stack(pot_val,0)
        pot_val=make_grid(pot_val,nrow=8,padding=5)
        self.writer.add_image('Pot_val',pot_val,epoch)


class AverageMeter(object):
    "help computing and storing the metrics"
    def __init__(self):
        self.val=0
        self.avg=0
        self.sum=0
        self.count=0

    def update(self,value,count=1):
        self.val=value
        self.sum+=value*count
        self.count+=count
        self.avg=self.sum/self.count

    @property
    def value(self):
        return self.val

    def average(self):
        return np.round(self.avg,5)

def batch_pix_accuracy(predict, target, labeled):
    pixel_labeled = labeled.sum()
    pixel_correct = ((predict == target) * labeled).sum()
    assert pixel_correct <= pixel_labeled, "Correct area should be smaller than Labeled"
    return pixel_correct.cpu().numpy(), pixel_labeled.cpu().numpy()

def batch_intersection_union(predict, target, num_class, labeled):
    predict = predict * labeled.long()
    intersection = predict * (predict == target).long()

    area_inter = torch.histc(intersection.float(), bins=num_class, max=num_class, min=1)
    area_pred = torch.histc(predict.float(), bins=num_class, max=num_class, min=1)
    area_lab = torch.histc(target.float(), bins=num_class, max=num_class, min=1)
    area_union = area_pred + area_lab - area_inter
    assert (area_inter <= area_union).all(), "Intersection area should be smaller than Union area"
    return area_inter.cpu().numpy(), area_union.cpu().numpy()

def Get_Segmetrics(output, target, num_class):
    predict = output + 1
    target = target +1

    labeled = (target > 0) * (target <= num_class)
    correct, num_labeled = batch_pix_accuracy(predict, target, labeled)
    inter, union = batch_intersection_union(predict, target, num_class, labeled)
    return [np.round(inter, 5), np.round(union, 5), np.round(correct, 5), np.round(num_labeled, 5)]

