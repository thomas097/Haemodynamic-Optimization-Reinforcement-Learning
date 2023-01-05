import numpy as np
import torch


class CosineWarmupScheduler:
    def __init__(self, warmup, max_lrate, max_epochs):
        """
        Cosine-based learning rate regime with warm-up
        :param warmup:      Number of warm-up epochs
        :param max_lrate:   Maximum learning rate
        :param max_epochs:  Number of epochs
        """
        print('\nUsing CosineWarmupScheduler')
        self._warmup = warmup
        self._max_lrate = max_lrate
        self._max_epochs = max_epochs

    def set(self, optimizer, epoch):
        """
        Set optimizer learning rate according to scheduler
        :param optimizer:  Torch Optimizer instance
        :param epoch:      Epoch number <= max_epochs
        :return:
        """
        if epoch < self._warmup:
            lrate = (epoch / self._warmup) * self._max_lrate
        else:
            lrate = self._max_lrate * 0.5 * (1 + np.cos(np.pi * (epoch - self._warmup) / (self._max_epochs - self._warmup)))

        for g in optimizer.param_groups:
            g['lr'] = lrate