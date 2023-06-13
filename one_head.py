
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from pytorch_metric_learning.losses import NTXentLoss

from tqdm import tqdm

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score

from dataset import tep
from model import encoder, linear
from utils import left2right as augment
# from utils import left2right, noise

# hparams
epochs = 100
batch_size = 64
lr = 5e-6
weight_decay = 1e-0
tau = 0.1

# dataset
train = tep()
val = tep(test = True)

train_loader = DataLoader(train, batch_size = batch_size, shuffle = True)
val_loader = DataLoader(val, batch_size = batch_size)

# encoder
model = encoder().cuda()
optimizer = optim.Adam(model.parameters(), lr = lr, weight_decay = weight_decay)
criterion = NTXentLoss(temperature = tau)

# train encoder
log = []
for epoch in tqdm(range(epochs)):
    model.train()
    loss_train = 0
    for x, _ in train_loader:
        optimizer.zero_grad()
        # da
        x_sim = augment(x)
        # pass
        latent = model.forward(x.float().cuda())
        latent_sim = model.forward(x_sim.float().cuda())
        embeddings = torch.cat((latent, latent_sim))
        # indices
        indices = torch.arange(0, latent.size(0)).cuda()
        labels = torch.cat((indices, indices))
        # loss
        loss = criterion(embeddings, labels)
        loss.backward()
        loss_train += loss.item()
        optimizer.step()
    
    model.eval()
    loss_eval = 0
    for x, _ in val_loader:
        # da
        x_sim = augment(x)
        # pass
        latent = model.forward(x.float().cuda())
        latent_sim = model.forward(x_sim.float().cuda())
        embeddings = torch.cat((latent, latent_sim))
        # indices
        indices = torch.arange(0, latent.size(0)).cuda()
        labels = torch.cat((indices, indices))
        # loss
        loss = criterion(embeddings, labels)
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
for epoch in tqdm(range(100)):
    clf.train()
    loss_train = 0
    for x, y in train_loader:
        optimizer.zero_grad()
        # embedding
        latent = model.forward(x.float().cuda())
        yhat = clf.forward(latent)
        # loss
        loss = criterion(yhat, y.float().cuda())
        loss.backward()
        loss_train += loss.item()
        optimizer.step()
    
    clf.eval()
    loss_eval = 0
    for x, y in val_loader:
        # embedding
        latent = model.forward(x.float().cuda())
        yhat = clf.forward(latent)
        # loss
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
