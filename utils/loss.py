import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn





def get_weights(target):
    t_np = target.view(-1).data.cpu().numpy()

    classes, counts = np.unique(t_np, return_counts=True)
    cls_w = np.median(counts) / counts
    # cls_w = class_weight.compute_class_weight('balanced', classes, t_np)

    weights = np.ones(7)
    weights[classes] = cls_w
    return torch.from_numpy(weights).float().cuda()

# class DiceLoss(nn.Module):
#     def __init__(self, smooth=1., ignore_index=255):
#         super(DiceLoss, self).__init__()
#         self.ignore_index = ignore_index
#         self.smooth = smooth
#
#     def forward(self, output, target):
#         if self.ignore_index not in range(target.min(), target.max()):
#             if (target == self.ignore_index).sum() > 0:
#                 target[target == self.ignore_index] = target.min()
#         target = make_one_hot(target.unsqueeze(dim=1), classes=output.size()[1])
#         output = F.softmax(output, dim=1)
#         output_flat = output.contiguous().view(-1)
#         target_flat = target.contiguous().view(-1)
#         intersection = (output_flat * target_flat).sum()
#         loss = 1 - ((2. * intersection + self.smooth) /
#                     (output_flat.sum() + target_flat.sum() + self.smooth))
#         return loss


class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=None, ignore_index=255, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.size_average = size_average
        self.CE_loss = nn.CrossEntropyLoss(reduce=False, ignore_index=ignore_index, weight=alpha)

    def forward(self, output, target):
        logpt = self.CE_loss(output, target)
        pt = torch.exp(-logpt)
        loss = ((1 - pt) ** self.gamma) * logpt
        if self.size_average:
            return loss.mean()
        return loss.sum()

