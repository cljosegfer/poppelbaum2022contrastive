#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  1 14:59:16 2023

@author: jose
"""

import os
import wget
import zipfile

from utils import moving_window

data_path = 'data'
if not os.path.exists(data_path):
    os.mkdir(data_path)
url = 'http://web.mit.edu/braatzgroup/TE_process.zip'
file = 'TE_process.zip'
folder = file[:-4]
if not os.path.exists(os.path.join(data_path, file)):
    wget.download(url, out = data_path)
if not os.path.exists(os.path.join(data_path, folder)):
    with zipfile.ZipFile(os.path.join(data_path, file), 'r') as z:
        z.extractall(data_path)

# d00 is special
i = 0
train_path = os.path.join(data_path, folder, 'd{:02}.dat'.format(i))
moving_window(train_path, d00_train = True)
test_path = os.path.join(data_path, folder, 'd{:02}_te.dat'.format(i))
moving_window(test_path)
for i in range(1, 22):
    case = 'd{:02}'.format(i)
    train_path = os.path.join(data_path, folder, 'd{:02}.dat'.format(i))
    moving_window(train_path)
    test_path = os.path.join(data_path, folder, 'd{:02}_te.dat'.format(i))
    moving_window(test_path, start = 160)
