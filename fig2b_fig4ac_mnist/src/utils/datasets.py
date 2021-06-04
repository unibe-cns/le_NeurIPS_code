#!/usr/bin/env python3

import numpy as np
import torch
from torch.utils.data import Dataset

import torchvision.datasets as datasets

class MnistDataset(Dataset):
    def __init__(self, which='train', num_classes=10, n_samples=-1, zero_at=0, one_at=1):
        self.cs = []
        self.vals = []
        self.class_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

        # ensure valid parameters
        assert 2 <= num_classes <= 10
        assert 100 <= n_samples <= 5000 or n_samples == -1

        if which == 'train':
            self.data = datasets.MNIST(root='../data/mnist', train=True, download=True, transform=None)
            if n_samples == -1:
                n_samples = 5000
            for i in range(0, 10 * n_samples):
                if self.data.targets[i] > num_classes - 1:
                    continue
                cs_flat = np.zeros(num_classes)
                cs_flat[self.data.targets[i]] = 1
                cs_flat = zero_at + cs_flat * (one_at - zero_at)
                self.cs.append(cs_flat)
                dat_flat = self.data.data[i].flatten().float()
                dat_flat /= 256.
                dat_flat = zero_at + dat_flat * (one_at - zero_at)
                self.vals.append(dat_flat)
        elif which == 'val':
            self.data = datasets.MNIST(root='../data/mnist', train=True, download=True, transform=None)
            if n_samples == -1:
                n_samples = 1000
            for i in range(50  * n_samples, 60 * n_samples):
                if self.data.targets[i] > num_classes - 1:
                    continue
                cs_flat = np.zeros(num_classes)
                cs_flat[self.data.targets[i]] = 1
                cs_flat = zero_at + cs_flat * (one_at - zero_at)
                self.cs.append(cs_flat)
                dat_flat = self.data.data[i].flatten().float()
                dat_flat /= 256.
                dat_flat = zero_at + dat_flat * (one_at - zero_at)
                self.vals.append(dat_flat)
        elif which == 'test':
            self.data = datasets.MNIST(root='../data/mnist', train=False, download=True, transform=None)
            if n_samples == -1:
                n_samples = 1000
            for i in range(10 * n_samples):
                if self.data.targets[i] > num_classes - 1:
                    continue
                cs_flat = np.zeros(num_classes)
                cs_flat[self.data.targets[i]] = 1
                cs_flat = zero_at + cs_flat * (one_at - zero_at)
                self.cs.append(cs_flat)
                dat_flat = self.data.data[i].flatten().float()
                dat_flat /= 256.
                dat_flat = zero_at + dat_flat * (one_at - zero_at)
                self.vals.append(dat_flat)

    def __getitem__(self, key):
        return self.vals[key], self.cs[key]

    def __setitem__(self, key, value):
        self.__dict__[key] = value

    def __len__(self):
        return len(self.cs)
