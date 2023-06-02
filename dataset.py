
import torch
from torch.utils.data import Dataset
import torch.nn as nn

import numpy as np
import os

class tep(Dataset):
    def __init__(self, test = False):
        self.X = None
        self.y = None
        self._read(test)
    
    def _read(self, test = False, data_path = 'data/TE_process'):
        L = 52
        T = 100
        self.X = np.empty(shape = [0, L, T])
        self.y = np.empty(shape = [0])
        for i in range(22):
            path = os.path.join(data_path, 'd{:02}.npy'.format(i))
            if test:
                path = os.path.join(data_path, 'd{:02}_te.npy'.format(i))
            X_case = np.load(path)
            y_case = i * np.ones(shape = X_case.shape[0])
            self.X = np.concatenate([self.X, X_case])
            self.y = np.concatenate([self.y, y_case])
        self.y = nn.functional.one_hot(torch.from_numpy(self.y.astype('int64')))
    
    def __len__(self):
        assert self.X.shape[0] == self.y.shape[0]
        return self.X.shape[0]
    
    def __getitem__(self, idx):
        return self.X[idx, :, :], self.y[idx]
