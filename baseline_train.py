
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

from tqdm import tqdm

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score

from dataset import tep
from model import baseline

# hparams
epochs = 300
batch_size = 64
lr = 4e-6
weight_decay = 1e-5

# dataset
# train, val = random_split(tep(), [0.7, 0.3])
data = tep()
train, val = random_split(data, [int(data.__len__() * i) for i in [0.7, 0.3]])

train_loader = DataLoader(train, batch_size = batch_size, shuffle = True)
val_loader = DataLoader(val, batch_size = val.__len__())

# model
model = baseline().cuda()
optimizer = optim.Adam(model.parameters(), lr = lr, weight_decay = weight_decay)
criterion = nn.BCEWithLogitsLoss()

# train
log = []
for epoch in tqdm(range(epochs)):
    model.train()
    loss_train = 0
    for x, y in train_loader:
        optimizer.zero_grad()
        yhat = model.forward(x.float().cuda())
        loss = criterion(yhat, y.float().cuda())
        loss.backward()
        optimizer.step()
        loss_train += loss.item()

    model.eval()
    loss_eval = 0
    with torch.no_grad():
        for x, y in val_loader:
            yhat = model.forward(x.float().cuda())
            loss = criterion(yhat, y.float().cuda())
            loss_eval += loss.item()
    log.append([loss_train / len(train_loader), loss_eval / len(val_loader)])

print(lr, weight_decay)
print(loss_train / len(train_loader), loss_eval / len(val_loader))

# plot
log = np.array(log)
plt.figure()
plt.plot(log[:, 0])
plt.plot(log[:, 1])
plt.savefig('output/train_log.png')
plt.close()

# eval
true = torch.argmax(y.cpu(), dim = 1)
labels = torch.argmax(yhat.cpu(), dim = 1)
print('acc: {}, f1: {}'.format(accuracy_score(true, labels), f1_score(true, labels, average = 'micro')))

# save
torch.save({
  'epoch': epochs,
  'state_dict': model.state_dict(),
  'eval_loss': loss.item(),
  }, 'output/baseline.pth.tar')
