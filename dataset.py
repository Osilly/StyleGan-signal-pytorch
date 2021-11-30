#!/usr/bin/env python
# coding: utf-8

import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import maxabs_scale
import os


class GetData:
    def __init__(self, path, signal_size=4096):
        real_classes = [d for d in os.listdir(path) if
                        os.path.isfile(os.path.join(path, d))]
        real_classes.sort(key=lambda x: int(x[0:-4]))
        real_signal = []
        for file_path in real_classes:
            real_signal.append(np.loadtxt(os.path.join(path, file_path)))

        real_signal = np.array(real_signal)
        real_signal = np.pad(real_signal, ((0, 0), (0, signal_size - real_signal.shape[-1])),
                             'constant', constant_values=(0, 0))

        real_signal = maxabs_scale(real_signal, axis=1)
        self.real_signal = real_signal[:, np.newaxis, np.newaxis, :]

    def get_data(self):
        return self.real_signal


class GetDataset(Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform

    def __getitem__(self, index):
        data = torch.Tensor(self.data[index])
        if self.transform is not None:
            data = self.transform(data)
        data = data.squeeze(1)
        return data

    def __len__(self):
        return len(self.data)
