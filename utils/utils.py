import os
import numpy as np
import PIL

import torch
from torch import nn
import torch.nn.init as initer
from torch.optim.lr_scheduler import _LRScheduler
import os
import numpy as np
import PIL.Image as Image





class Poly(_LRScheduler):
    def __init__(self, optimizer, num_epochs, iters_per_epoch=0, warmup_epochs=0, last_epoch=-1):
        self.iters_per_epoch = iters_per_epoch
        self.cur_iter = 0
        self.N = num_epochs * iters_per_epoch
        self.warmup_iters = warmup_epochs * iters_per_epoch
        super(Poly, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        T = self.last_epoch * self.iters_per_epoch + self.cur_iter
        factor =  pow((1 - 1.0 * T / self.N), 0.9)
        if self.warmup_iters > 0 and T < self.warmup_iters:
            factor = 1.0 * T / self.warmup_iters

        self.cur_iter %= self.iters_per_epoch
        self.cur_iter += 1
        return [base_lr * factor for base_lr in self.base_lrs]

def colorization(mask, palette):
    zero_pad = 256 * 3 - len(palette)
    for i in range(zero_pad):
                    palette.append(0)
    #print(mask.shape)
    deseg_mask = PIL.Image.fromarray(mask.astype(np.uint8)).convert('P')
    deseg_mask.putpalette(palette)
    #print(np.array(deseg_mask))
    return deseg_mask

def Check_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def Free_port():
    import socket
    sock=socket.socket(socket.AF_INET,socket.SOCK_STREAM)

    sock.bind(("",0))
    port=sock.getsockname()[1]
    sock.close()
    return port



if __name__ == "__main__":
    import torchvision
    import torch
    import matplotlib.pylab as plt

    resnet = torchvision.models.resnet34()
    params = {
        "lr": 0.01,
        "weight_decay": 0.001,
        "momentum": 0.9
    }
    optimizer = torch.optim.SGD(params=resnet.parameters(), **params)

    epochs = 2
    iters_per_epoch = 100
    lrs = []
    mementums = []
    lr_scheduler = Poly(optimizer, epochs, iters_per_epoch)
    #lr_scheduler = Poly(optimizer, epochs, iters_per_epoch)

    for epoch in range(epochs):
        for i in range(iters_per_epoch):
            lr_scheduler.step(epoch=epoch)
            lrs.append(optimizer.param_groups[0]['lr'])
            mementums.append(optimizer.param_groups[0]['momentum'])

    plt.ylabel("learning rate")
    plt.xlabel("iteration")
    plt.plot(lrs)
    plt.show()

    plt.ylabel("momentum")
    plt.xlabel("iteration")
    plt.plot(mementums)
    plt.show()