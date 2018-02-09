import numpy as np
import pandas as pd
from tqdm import tqdm

import matplotlib.pyplot as plt
plt.style.use('seaborn')

import torch as T
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import torch.optim as optim

from .. import util


class UNetTrainer(object):
    def __init__(self, model, ckpt_dir):
        super().__init__()
        self.model = model
        self.n_epochs = 250
        self.criterion = nn.BCELoss()
        self.optimizer = optim.Adam(self.model.parameters())
        self.ckpt_dir = ckpt_dir
        self.log = None

        print('CKPT:', self.ckpt_dir)

    def __train(self, ep, msg, pbar):
        batch_size = 10
        msg.update({'loss': 0.0})
        self.model.train()
        for i, (xb, yb) in enumerate(
                util.make_batch(self.xt, self.yt, bs=batch_size)):
            xb = Variable(xb.cuda(), requires_grad=True)
            yb = Variable(yb.cuda(), requires_grad=False)

            self.optimizer.zero_grad()
            out = self.model(xb)
            loss = self.criterion(out, yb)
            loss.backward()
            self.optimizer.step()

            msg['loss'] = (msg['loss'] * i + loss.data[0]) / (i + 1)
            pbar.set_postfix(**msg)
            pbar.update(batch_size)

    def __valid(self, ep, msg, pbar):
        batch_size = 5
        msg.update({'val_loss': 0.0})
        self.model.eval()
        for i, (xb, yb) in enumerate(
                util.make_batch(self.xv, self.yv, bs=batch_size)):
            xb = Variable(xb.cuda(), requires_grad=False)
            yb = Variable(yb.cuda(), requires_grad=False)

            out = self.model(xb)
            loss = self.criterion(out, yb)
            msg['val_loss'] = (msg['val_loss'] * i + loss.data[0]) / (i + 1)
        pbar.set_postfix(**msg)

    def __log(self, ep, msg, pbar):
        if ep == 0 or msg['val_loss'] < self.log['val_loss'].min():
            with (self.ckpt_dir / 'model.pth').open('wb') as f:
                T.save(self.model, f)

        if self.log is None:
            self.log = pd.DataFrame([msg])
        else:
            self.log = self.log.append(msg, ignore_index=True)
        self.log.to_csv(str(self.ckpt_dir / 'log.csv'), index_label='epoch')

        fig, ax = plt.subplots(dpi=150)
        self.log.plot(kind='line', ax=ax)
        ax.set_xlabel('epoch')
        ax.set_ylabel('loss')
        fig.tight_layout()
        fig.savefig(str(self.ckpt_dir / 'loss.png'))
        plt.close()

    def __vis(self, ep, msg, pbar):
        epoch_dir = self.ckpt_dir / f'{ep:03d}'
        epoch_dir.mkdir()

        xs = Variable(self.xvis.cuda(), requires_grad=False)
        yp = self.model(xs).cpu().data.numpy()

        xvis = np.transpose(self.xvis, [0, 2, 3, 1])
        yvis = np.transpose(self.yvis, [0, 2, 3, 1])
        yp = np.transpose(yp, [0, 2, 3, 1])

        for i, (x, yt, yp) in enumerate(zip(xvis, yvis, yp)):
            fig, axes = plt.subplots(nrows=2, ncols=2, dpi=150)
            axes[0, 0].imshow(x)
            axes[0, 1].imshow(yt[..., 0], cmap='gray')
            axes[1, 0].imshow(yp[..., 0], cmap='jet')
            axes[1, 2].imshow(yp[..., 1], cmap='jet')
            for r in range(2):
                for c in range(2):
                    axes[r, c].axis('off')
            fig.tight_layout()
            fig.savefig(str(epoch_dir / f'{i:03d}.jpg'))
            plt.close()

    def fit(self, xt, yt, xv, yv):
        '''
            xt, yt, xv, yv should be ndarray with shape (N, C, W, H)
        '''
        assert len(xt) == len(yt)
        assert len(xv) == len(yv)

        n_samples = len(xt)
        (self.xt, self.yt) = (T.from_numpy(xt), T.from_numpy(yt))
        (self.xv, self.yv) = (T.from_numpy(xv), T.from_numpy(yv))
        self.xvis = T.cat((self.xt[:5], self.xv[:5]), dim=0)
        self.yvis = T.cat((self.yt[:5], self.yv[:5]), dim=0)

        self.model = self.model.cuda()
        for ep in range(self.n_epochs):
            with tqdm(
                    total=n_samples, desc=f'Epoch {ep:03d}',
                    ascii=True) as pbar:
                msg = dict()
                self.__train(ep, msg, pbar)
                self.__valid(ep, msg, pbar)
                self.__log(ep, msg, pbar)
                self.__vis(ep, msg, pbar)