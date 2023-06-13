#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  1 16:00:31 2023

@author: jose
"""

import pandas as pd
import numpy as np
import torch

def moving_window(path, T = 100, start = 0, d00_train = False):
    df = pd.read_csv(path, delim_whitespace = True, header = None)
    X = df.values
    if d00_train:
        X = X.T
    N, L = X.shape
    N = N - start
    assert L == 52
    processed = np.zeros(shape = [N - T, L, T])
    for i in range(start, N - T):
        signal = X[i:i+T, :]
        processed[i, :, :] = signal.T
    np.save(path[:-3] + 'npy', processed)
    # print(path, i, processed.shape)
    
def left2right(X, T = 100):
    assert X.shape[2] == T
    Y = np.eye(T)[::-1]
    return np.matmul(X, Y)

def noise(X, lambda_ = 0.1):
    Sigma = torch.std(X, axis = 0)
    R = np.random.uniform(-1, 1, size = X.shape)
    return X + torch.from_numpy(R) * lambda_ * Sigma
