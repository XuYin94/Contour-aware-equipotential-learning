import torch
import time
import numpy as np
from torchvision.utils import make_grid
from torchvision import transforms
from utils import transforms as local_transforms
from base import BaseTrainer, DataPrefetcher
from utils.helpers import colorize_mask
from utils.metrics import eval_metrics, AverageMeter
from PIL import Image
import time
import cv2
from tqdm import tqdm

import torch.nn.functional as F
class Trainer(BaseTrainer):
    def __init__(self, model, loss, resume, config, train_loader, val_loader=None, train_logger=None, prefetch=True):
        super(Trainer, self).__init__(model, loss, resume, config, train_loader, val_loader, train_logger)

        self.wrt_mode, self.wrt_step = 'train_', 0
        self.log_step = config['trainer'].get('log_per_iter', int(np.sqrt(self.train_loader.batch_size)))
        if config['trainer']['log_per_iter']: self.log_step = int(self.log_step / self.train_loader.batch_size) + 1

        self.num_classes = self.train_loader.dataset.num_classes

        # TRANSORMS FOR VISUALIZATION
        self.restore_transform = transforms.Compose([
            local_transforms.DeNormalize(self.train_loader.MEAN, self.train_loader.STD),
            transforms.ToPILImage()])
        self.viz_transform = transforms.Compose([
            transforms.Resize((400, 400)),
            transforms.ToTensor()])

        self.restore_transform = transforms.Compose([
            local_transforms.DeNormalize(self.train_loader.MEAN, self.train_loader.STD),
            transforms.ToPILImage()])

        self.kernel_transform = transforms.Compose([local_transforms.kerel_viz(),
            transforms.ToPILImage()])
        self.dis_transform = transforms.Compose([local_transforms.dis_viz(),
            transforms.ToPILImage()])

        self.distance_transform = transforms.Compose([local_transforms.sdf_viz(),
            transforms.ToPILImage()])

        self.learned_transform = local_transforms.Co_viz()
        self.to_PIL=transforms.Compose([transforms.ToPILImage()])

        self.viz_transform = transforms.Compose([
            transforms.Resize((400, 400)),
            transforms.ToTensor()])

        # self.dis_transform = transforms.Compose([local_transforms.kerel_viz(),
        #     transforms.ToPILImage()])

        if self.device == torch.device('cpu'): prefetch = False
        if prefetch:
            self.train_loader = DataPrefetcher(train_loader, device=self.device)
            self.val_loader = DataPrefetcher(val_loader, device=self.device)

        torch.backends.cudnn.benchmark = True

    def _train_epoch(self, epoch):
        self.logger.info('\n')

        self.model.train()
        if self.config['arch']['args']['freeze_bn']:
            if isinstance(self.model, torch.nn.DataParallel):
                self.model.module.freeze_bn()
            else:
                self.model.freeze_bn()
        self.wrt_mode = 'train'

        tic = time.time()
        self._reset_metrics()
        tbar = tqdm(self.train_loader, ncols=130)
        for batch_idx, (data, target,distance) in enumerate(tbar):
            self.data_time.update(time.time() - tic)
            # data, target = data.to(self.device), target.to(self.device)


            # LOSS & OPTIMIZE
            self.kernel_optimizer.zero_grad()
            self.back_optimizer.zero_grad()
            output,distance_loss= self.model(data,distance)
            #print(output.shape)
            assert output.size()[2:] == target.size()[1:]
            assert output.size()[1] == self.num_classes
            seg_loss = self.loss(output, target)
            loss=5*seg_loss+distance_loss
            loss.backward()
            self.kernel_optimizer.step()
            self.back_optimizer.step()

            self.kernel_lr_scheduler.step(epoch=epoch - 1)
            self.back_lr_scheduler.step(epoch=epoch - 1)

            self.total_loss.update(loss.item())

            # measure elapsed time
            self.batch_time.update(time.time() - tic)
            tic = time.time()

            # LOGGING & TENSORBOARD
            if batch_idx % self.log_step == 0:
                self.wrt_step = (epoch - 1) * len(self.train_loader) + batch_idx
                self.writer.add_scalar(f'{self.wrt_mode}/loss', loss.item(), self.wrt_step)
                self.writer.add_scalar(f'{self.wrt_mode}/seg_loss', seg_loss, self.wrt_step)
                self.writer.add_scalar(f'{self.wrt_mode}/dis_loss', distance_loss, self.wrt_step)
            # FOR EVAL
            seg_metrics = eval_metrics(output, target, self.num_classes)
            self._update_seg_metrics(*seg_metrics)
            pixAcc, mIoU, _ = self._get_seg_metrics().values()

            # PRINT INFO
            tbar.set_description('TRAIN ({}) | Seg_Loss: {:.3f} | dis_Loss: {:.3f} |Acc {:.2f} mIoU {:.2f} | B {:.2f} D {:.2f} |'.format(
                epoch, 5*seg_loss,distance_loss,
                pixAcc, mIoU,
                self.batch_time.average, self.data_time.average))

        # METRICS TO TENSORBOARD
        seg_metrics = self._get_seg_metrics()
        for k, v in list(seg_metrics.items())[:-1]:
            self.writer.add_scalar(f'{self.wrt_mode}/{k}', v, self.wrt_step)


        for i, opt_group in enumerate(self.kernel_optimizer.param_groups):
            self.writer.add_scalar(f'{self.wrt_mode}/kernel_lr_{i}', opt_group['lr'], self.wrt_step)
            # self.writer.add_scalar(f'{self.wrt_mode}/Momentum_{k}', opt_group['momentum'], self.wrt_step)
        for i, opt_group in enumerate(self.back_optimizer.param_groups):
            self.writer.add_scalar(f'{self.wrt_mode}/back_lr_{i}', opt_group['lr'], self.wrt_step)
        # RETURN LOSS & METRICS
        log = {'loss': self.total_loss.average,
               **seg_metrics}

        # if self.lr_scheduler is not None: self.lr_scheduler.step()
        return log

    def _valid_epoch(self, epoch):
        if self.val_loader is None:
            self.logger.warning('Not data loader was passed for the validation step, No validation is performed !')
            return {}
        self.logger.info('\n###### EVALUATION ######')

        self.model.eval()
        self.wrt_mode = 'val'

        self._reset_metrics()
        tbar = tqdm(self.val_loader, ncols=130)
        with torch.no_grad():
            val_visual = []
            kernel_visual=[]
            for batch_idx, (data, target,distance) in enumerate(tbar):
                output,distance_loss = self.model(data,distance)
                kernel,coarse_estimation,learned_sdf=self.model.module.kernel_visualization(data)

                seg_loss = self.loss(output, target)
                loss=5*seg_loss+distance_loss
                self.total_loss.update(loss.item())

                seg_metrics = eval_metrics(output, target, self.num_classes)
                self._update_seg_metrics(*seg_metrics)

                # LIST OF IMAGE TO VIZ (15 images)
                if len(val_visual) < 15:
                    target_np = target.data.cpu().numpy()
                    kernel_np=kernel[:,1,:,:,].data.cpu().numpy()
                    output_np = output.data.max(1)[1].cpu().numpy()
                    #print(conved.shape)

                    learned_sdf_np=learned_sdf.data.cpu().numpy()##B*k*W*H
                    target_sdf_np=distance.data.cpu().numpy()##B*k*W*H

                    coarse_estimation_np=coarse_estimation[:,1,:,:,].data.cpu().numpy()

                    val_visual.append([data[0].data.cpu(), target_np[0], output_np[0]])## RGB, Segmentation label, segmentation, target_sdf.
                    kernel_visual.append([data[0].data.cpu(),kernel_np[0],
                                          coarse_estimation_np[0],learned_sdf_np[0],target_sdf_np[0]])
                # PRINT INFO
                pixAcc, mIoU, _ = self._get_seg_metrics().values()
                tbar.set_description('EVAL ({}) | Seg_Loss: {:.3f}, Dis_Loss: {:.3f},PixelAcc: {:.2f}, Mean IoU: {:.2f} |'.format(epoch,
                                                                                                      seg_loss,distance_loss,
                                                                                                             pixAcc,
                                                                                                             mIoU))

            # WRTING & VISUALIZING THE MASKS
            val_img = []
            palette = self.train_loader.dataset.palette
            for rgb, label, seg in val_visual:
                rgb= self.restore_transform(rgb)
                label, seg = colorize_mask(label, palette), colorize_mask(seg, palette)
                rgb, label, seg= rgb.convert('RGB'), label.convert('RGB'), seg.convert('RGB')
                [rgb, label, seg] = [self.viz_transform(x) for x in [rgb, label, seg]]
                val_img.extend([rgb, label, seg])
            kernel_img = []

            for rgb, kernel, pro,learned_sdf,sdf in kernel_visual:## RGB, kerenl,coarse_prediction, learned_field, target_field
                rgb,kernel,pro=self.restore_transform(rgb),self.kernel_transform(kernel),self.kernel_transform(pro)
                learned_sdf, sdf=self.learned_transform(learned_sdf,sdf)
                learned_sdf,sdf=self.to_PIL(learned_sdf),self.to_PIL(sdf)

                rgb, kernel, pro,learned_sdf,sdf= rgb.convert('RGB'), kernel.convert('RGB'), pro.convert('RGB'),\
                                                      learned_sdf.convert('RGB'),sdf.convert('RGB')
                [rgb, kernel, pro,learned_sdf,sdf] = [self.viz_transform(x) for x in [rgb, kernel, pro,learned_sdf,sdf]]
                kernel_img.extend([rgb, kernel, pro,learned_sdf,sdf])

            val_img = torch.stack(val_img, 0)
            val_img = make_grid(val_img.cpu(), nrow=3, padding=5)

            kernel_img = torch.stack(kernel_img, 0)
            kernel_img = make_grid(kernel_img.cpu(), nrow=5, padding=5)
            self.writer.add_image(f'{self.wrt_mode}/inputs_targets_predictions', val_img, self.wrt_step)
            self.writer.add_image(f'{self.wrt_mode}/kernel_predictions', kernel_img, self.wrt_step)
            # METRICS TO TENSORBOARD
            self.wrt_step = (epoch) * len(self.val_loader)
            self.writer.add_scalar(f'{self.wrt_mode}/loss', loss.item(), self.wrt_step)
            self.writer.add_scalar(f'{self.wrt_mode}/seg_loss',5*seg_loss,self.wrt_step)
            self.writer.add_scalar(f'{self.wrt_mode}/dis_loss',distance_loss,self.wrt_step)
            seg_metrics = self._get_seg_metrics()
            for k, v in list(seg_metrics.items())[:-1]:
                self.writer.add_scalar(f'{self.wrt_mode}/{k}', v, self.wrt_step)

            log = {
                'val_loss': self.total_loss.average,
                **seg_metrics
            }

        return log

    def _reset_metrics(self):
        self.batch_time = AverageMeter()
        self.data_time = AverageMeter()
        self.total_loss = AverageMeter()
        self.total_inter, self.total_union = 0, 0
        self.total_correct, self.total_label = 0, 0

    def _update_seg_metrics(self, correct, labeled, inter, union):
        self.total_correct += correct
        self.total_label += labeled
        self.total_inter += inter
        self.total_union += union

    def _get_seg_metrics(self):
        pixAcc = 1.0 * self.total_correct / (np.spacing(1) + self.total_label)
        IoU = 1.0 * self.total_inter / (np.spacing(1) + self.total_union)
        mIoU = IoU.mean()
        return {
            "Pixel_Accuracy": np.round(pixAcc, 3),
            "Mean_IoU": np.round(mIoU, 3),
            "Class_IoU": dict(zip(range(self.num_classes), np.round(IoU, 3)))
        }