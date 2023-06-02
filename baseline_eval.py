
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from sklearn.metrics import accuracy_score, f1_score

from dataset import tep
from model import baseline

# data
test = tep(test = True)
test_loader = DataLoader(test, batch_size = test.__len__())

# model
model = baseline().cuda()
checkpoint = torch.load('output/baseline.pth.tar')
model.load_state_dict(checkpoint['state_dict'])
criterion = nn.BCEWithLogitsLoss()

# eval
model.eval()
with torch.no_grad():
    for x, y in test_loader:
        yhat = model.forward(x.float().cuda())
        loss = criterion(yhat, y.float().cuda())
print(loss.item())
true = torch.argmax(y.cpu(), dim = 1)
labels = torch.argmax(yhat.cpu(), dim = 1)
print('acc: {}, f1: {}'.format(accuracy_score(true, labels), f1_score(true, labels, average = 'micro')))
