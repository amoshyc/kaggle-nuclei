import math
from collections import OrderedDict
from tqdm import tqdm

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 8, (3, 3), padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, (3, 3), padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2), )
        self.center = nn.Sequential(
            nn.Conv2d(16, 16, (3, 3), padding=1),
            nn.ReLU(), )
        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(16, 8, (3, 3), padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(8, 1, (3, 3), padding=1),
            nn.Sigmoid(), )

    def forward(self, x):
        x = self.encoder(x)
        x = self.center(x)
        x = self.decoder(x)
        return x


def split(xs, ys, batch_size):
    assert len(xs) == len(ys)
    n = len(xs)
    n_iter = math.ceil(n / batch_size)
    for i in range(n_iter):
        s = i * batch_size
        t = min(s + batch_size, n)
        yield i, xs[s:t], ys[s:t]


def train(net, xt, yt, xv, yv):
    batch_size = 30
    n_epochs = 250

    print(net)
    net = net.cuda()
    optimizer = optim.Adam(net.parameters())
    criterion = nn.BCELoss()

    for ep in range(n_epochs):
        pbar = tqdm(total=len(xt), desc=f'Epoch {ep:03d}', ascii=True)

        net.train()
        msg = OrderedDict({'loss': 0.0})
        for i, xb, yb in split(xt, yt, batch_size):
            xb = Variable(torch.from_numpy(xb).cuda(), requires_grad=True)
            yb = Variable(torch.from_numpy(yb).cuda(), requires_grad=False)

            optimizer.zero_grad()
            yp = net(xb)
            loss = criterion(yp, yb)
            loss.backward()
            optimizer.step()

            msg['loss'] = (msg['loss'] * i + loss.data[0]) / (i + 1)
            pbar.set_postfix(**msg)
            pbar.update(batch_size)

        net.eval()
        msg.update(OrderedDict({'val_loss': 0.0}))
        for i, xb, yb in split(xv, yv, batch_size):
            xb = Variable(torch.from_numpy(xb).cuda(), requires_grad=True)
            yb = Variable(torch.from_numpy(yb).cuda(), requires_grad=False)

            yp = net(xb)
            loss = heatmap_loss(yp, yb)
            
            msg['val_loss'] = (msg['val_loss'] * i + loss.data[0]) / (i + 1)
        pbar.set_postfix(**msg)
        pbar.close()
