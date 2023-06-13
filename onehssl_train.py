
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from info_nce import InfoNCE
from pytorch_metric_learning.losses import NTXentLoss

from tqdm import tqdm

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score

from dataset import tep
from model import encoder, linear, ContrastiveLoss
# from utils import noise as augment
from utils import left2right, noise

# hparams
epochs = 50
batch_size = 64
lr = 1e-5
weight_decay = 1e-5
epochs_pos = 100
tau = 0.1

# dataset
# data = tep()
# train, val = random_split(data, [int(data.__len__() * i) for i in [0.7, 0.3]])
train = tep()
val = tep(test = True)

train_loader = DataLoader(train, batch_size = batch_size, shuffle = True)
val_loader = DataLoader(val, batch_size = batch_size)

# encoder
model = encoder().cuda()
optimizer = optim.Adam(model.parameters(), lr = lr, weight_decay = weight_decay)
# criterion = InfoNCE(temperature = tau)
criterion = ContrastiveLoss(temperature = tau)

# train encoder
log = []
for epoch in tqdm(range(epochs)):
    model.train()
    loss_train = 0
    for x, _ in train_loader:
        optimizer.zero_grad()
        # x_sim = augment(x)
        x_sim = left2right(x)
        x = noise(x)
        latent = model.forward(x.float().cuda())
        latent_sim = model.forward(x_sim.float().cuda())
        loss = criterion(latent, latent_sim)
        loss.backward()
        optimizer.step()
        loss_train += loss.item()

    model.eval()
    loss_eval = 0
    with torch.no_grad():
        for x, _ in val_loader:
            # x_sim = augment(x)
            x_sim = left2right(x)
            x = noise(x)
            latent = model.forward(x.float().cuda())
            latent_sim = model.forward(x_sim.float().cuda())
            loss = criterion(latent, latent_sim)
            loss_eval += loss.item()
    log.append([loss_train / len(train_loader), loss_eval / len(val_loader)])

print(lr, weight_decay, tau)
print(loss_train / len(train_loader), loss_eval / len(val_loader))

# plot
log = np.array(log)
plt.figure()
plt.plot(log[:, 0])
plt.plot(log[:, 1])
plt.savefig('output/train_log.png')
plt.close()

# save
torch.save({
  'epoch': epochs,
  'state_dict': model.state_dict(),
  'eval_loss': loss.item(),
  }, 'output/encoder.pth.tar')

# linear
clf = linear().cuda()
optimizer = optim.Adam(clf.parameters(), lr = 1e-5, weight_decay = 1e-5)
criterion = nn.BCEWithLogitsLoss()
val_loader = DataLoader(val, batch_size = val.__len__())

# train linear
log = []
for epoch in tqdm(range(epochs_pos)):
    clf.train()
    loss_train = 0
    for x, y in train_loader:
        optimizer.zero_grad()
        latent = model.forward(x.float().cuda())
        yhat = clf.forward(latent)
        loss = criterion(yhat, y.float().cuda())
        loss.backward()
        optimizer.step()
        loss_train += loss.item()

    clf.eval()
    loss_eval = 0
    with torch.no_grad():
        for x, y in val_loader:
            latent = model.forward(x.float().cuda())
            yhat = clf.forward(latent)
            loss = criterion(yhat, y.float().cuda())
            loss_eval += loss.item()
    log.append([loss_train / len(train_loader), loss_eval / len(val_loader)])
print(loss_train / len(train_loader), loss_eval / len(val_loader))

# plot
log = np.array(log)
plt.figure()
plt.plot(log[:, 0])
plt.plot(log[:, 1])
plt.savefig('output/train_log2.png')
plt.close()

# eval
true = torch.argmax(y.cpu(), dim = 1)
labels = torch.argmax(yhat.cpu(), dim = 1)
print('acc: {}, f1: {}'.format(accuracy_score(true, labels), f1_score(true, labels, average = 'micro')))

# save
torch.save({
  'epoch': epochs,
  'state_dict': clf.state_dict(),
  'eval_loss': loss.item(),
  }, 'output/linear.pth.tar')
