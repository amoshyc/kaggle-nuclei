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
from ..loss.ae import tag_loss


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 8, (3, 3), padding=1),
            nn.ReLU(),
            nn.Conv2d(8, 8, (3, 3), padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, (3, 3), padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 16, (3, 3), padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.MaxPool2d(2),
        )
        self.center = nn.Sequential(
            nn.Conv2d(16, 32, (3, 3), padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, (3, 3), padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
        )
        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(32, 16, (3, 3), padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 16, (3, 3), padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(16, 8, (3, 3), padding=1),
            nn.ReLU(),
            nn.Conv2d(8, 8, (3, 3), padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(8),
            nn.Conv2d(8, 2, (3, 3), padding=1),
            nn.ReLU(),
            nn.Conv2d(2, 2, (3, 3), padding=1),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.center(x)
        x = self.decoder(x)
        heatmap = F.sigmoid(x[:, 0])
        tag_map = x[:, 1]
        return heatmap, tag_map


def split(xs, ys, ms, cs, batch_size):
    assert len(xs) == len(ys)
    assert len(ys) == len(ms)
    assert len(ms) == len(cs)
    n = len(xs)
    n_iter = math.ceil(n / batch_size)
    for i in range(n_iter):
        s = i * batch_size
        t = min(s + batch_size, n)
        yield i, xs[s:t], ys[s:t], ms[s:t], cs[s:t]


def train(net, xt, yt, mt, ct, xv, yv, mv, cv):
    batch_size = 30
    n_epochs = 250
    ckpt_dir = ckpt.new_dir()

    print(net)
    print('CKPT:', ckpt_dir)

    net = net.cuda()
    optimizer = optim.Adam(net.parameters())
    hm_loss = nn.BCELoss()


    def __train(ep, msg, pbar):
        net.train()
        msg.update({'loss': 0.0, 'loss_hm': 0.0, 'loss_tag': 0.0})
        for i, xb, yb, mb, cb in split(xt, yt, mt, ct, batch_size):
            xb = Variable(torch.from_numpy(xb).cuda(), requires_grad=True)
            yb = Variable(torch.from_numpy(yb).cuda(), requires_grad=False)

            optimizer.zero_grad()
            heatmap, tag_map = net(xb)

            loss_hm = hm_loss(heatmap, yb[:, 0, ...])
            loss_tag = tag_loss(tag_map, cb)
            loss = loss_hm + loss_tag
            loss.backward()
            optimizer.step()

            msg['loss'] = (msg['loss'] * i + loss.data[0]) / (i + 1)
            msg['loss_hm'] = (msg['loss_hm'] * i + loss_hm.data[0]) / (i + 1)
            msg['loss_tag'] = (msg['loss_tag'] * i + loss_tag.data[0]) / (i + 1)
            pbar.set_postfix(**msg)
            pbar.update(batch_size)

    def __valid(ep, msg, pbar):
        net.eval()
        msg.update({'val_loss': 0.0, 'val_loss_hm': 0.0, 'val_loss_tag': 0.0})
        for i, xb, yb, mb, cb in split(xv, yv, mv, cv, batch_size):
            xb = Variable(torch.from_numpy(xb).cuda(), requires_grad=True)
            yb = Variable(torch.from_numpy(yb).cuda(), requires_grad=False)

            heatmap, tag_map = net(xb)
            loss_hm = hm_loss(heatmap, yb[:, 0, ...])
            loss_tag = tag_loss(tag_map, cb)
            loss = loss_hm + loss_tag

            msg['val_loss'] = (msg['val_loss'] * i + loss.data[0]) / (i + 1)
            msg['val_loss_hm'] = (msg['val_loss_hm'] * i + loss_hm.data[0]) / (i + 1)
            msg['val_loss_tag'] = (msg['val_loss_tag'] * i + loss_tag.data[0]) / (i + 1)
        pbar.set_postfix(**msg)

    def __log(ep, msg, pbar):
        try:
            __log.log[ep] = msg
        except AttributeError:
            __log.log = {ep: msg}

        df = pd.DataFrame.from_dict(__log.log, orient='index')
        df.to_csv(str(ckpt_dir / 'log.csv'), index_label='epoch')

        fig, ax = plt.subplots(dpi=150)
        df.plot(kind='line', ax=ax)
        ax.set_xlabel('epoch')
        ax.set_ylabel('loss')
        fig.tight_layout()
        fig.savefig(str(ckpt_dir / 'loss.png'))
        plt.close()

    def __vis(ep, msg, pbar):
        epoch_dir = ckpt_dir / f'{ep:03d}'
        epoch_dir.mkdir()
        n_samples = 20
        xc, yc = xv[:n_samples], yv[:n_samples]
        xc_var = Variable(torch.from_numpy(xc).cuda(), requires_grad=False)
        heatmap, tag_map = net(xc_var)
        heatmap = heatmap.data.cpu().numpy()
        tag_map = tag_map.data.cpu().numpy()
        xc = np.transpose(xc, [0, 2, 3, 1])
        yc = np.transpose(yc, [0, 2, 3, 1])
        tag_map /= tag_map.max()
        for i in range(n_samples):
            fig, ax = plt.subplots(2, 2, dpi=100)
            ax[0, 0].imshow(xc[i])
            ax[0, 1].imshow(yc[i, ..., 0], cmap='gray')
            ax[1, 0].imshow(heatmap[i], cmap='jet')
            ax[1, 1].imshow(tag_map[i], cmap='viridis')
            for r in range(2):
                for c in range(2):
                    ax[r, c].axis('off')
            fig.tight_layout()
            fig.savefig(str(epoch_dir / f'vis{i:03d}.jpg'))
            plt.close()

    for ep in range(n_epochs):
        with tqdm(total=len(xt), desc=f'Epoch {ep:03d}', ascii=True) as pbar:
            msg = dict()
            __train(ep, msg, pbar)
            __valid(ep, msg, pbar)
            __log(ep, msg, pbar)
            __vis(ep, msg, pbar)