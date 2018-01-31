import math
from tqdm import tqdm

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from ..utils import ckpt
from ..utils import convert


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

    ckpt_dir = ckpt.new_dir()
    print('CKPT:', ckpt_dir)

    def __train(ep, msg, pbar):
        net.train()
        msg.update({'loss': 0.0})
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

    def __valid(ep, msg, pbar):
        net.eval()
        msg.update({'val_loss': 0.0})
        for i, xb, yb in split(xv, yv, batch_size):
            xb = Variable(torch.from_numpy(xb).cuda(), requires_grad=False)
            yb = Variable(torch.from_numpy(yb).cuda(), requires_grad=False)

            yp = net(xb)
            loss = criterion(yp, yb)

            msg['val_loss'] = (msg['val_loss'] * i + loss.data[0]) / (i + 1)
        pbar.set_postfix(**msg)

    def __vis(ep, msg, pbar):
        epoch_dir = ckpt_dir / f'{ep:03d}'
        epoch_dir.mkdir()
        xc, yc = xv[:20], yv[:20]
        xc_var = Variable(torch.from_numpy(xc).cuda(), requires_grad=False)
        yp = net(xc_var).data.cpu().numpy()
        xc = np.transpose(xc, [0, 2, 3, 1])
        yc = np.transpose(yc, [0, 2, 3, 1])
        yp = np.transpose(yp, [0, 2, 3, 1])
        for i in range(len(xc)):
            img = xc[i]
            truth = convert.colorize(yc[i], cmap='white')
            pred = convert.colorize(yp[i], cmap='orange')

            vis = np.hstack([img, truth, pred])
            vis = convert.to_pil_img(vis)
            vis.save(str(epoch_dir / f'{i:03d}.jpg'))

    def __plot(ep, msg, pbar):
        try:
            __plot.log[ep] = msg
        except AttributeError:
            __plot.log = {ep: msg}

        df = pd.DataFrame.from_dict(__plot.log, orient='index')
        df.to_csv(str(ckpt_dir / 'log.csv'), index_label='epoch')

        fig, ax = plt.subplots(dpi=150)
        df.plot(kind='line', ax=ax)
        ax.set_xlabel('epoch')
        ax.set_ylabel('loss')
        fig.tight_layout()
        fig.savefig(str(ckpt_dir / 'loss.png'))
        plt.close()

    for ep in range(n_epochs):
        with tqdm(total=len(xt), desc=f'Epoch {ep:03d}', ascii=True) as pbar:
            msg = dict()
            __train(ep, msg, pbar)
            __valid(ep, msg, pbar)
            __vis(ep, msg, pbar)
            __plot(ep, msg, pbar)