import numpy as np
import pandas as pd
from tqdm import tqdm
from skimage import color
from skimage import measure

import matplotlib.pyplot as plt
plt.style.use('seaborn')

import torch as T
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import torch.optim as optim

from .. import util
from . import predict


class Trainer(object):
    def __init__(self, model, ckpt_dir):
        super().__init__()
        self.model = model
        self.n_epochs = 250
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters())
        self.ckpt_dir = ckpt_dir
        self.log = None
        self.cmap = plt.cm.spectral

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
                T.save(self.model.state_dict(), f)

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

        xvis = np.transpose(self.xvis.numpy(), [0, 2, 3, 1])
        yvis = np.transpose(self.yvis.numpy(), [0, 2, 3, 1])
        yps = np.transpose(yp, [0, 2, 3, 1])

        for i, (x, yt, yp) in enumerate(zip(xvis, yvis, yps)):
            rgb_yt = self.cmap(yt[..., 0])[..., :3]
            rgb_yp = self.cmap(yp[..., 0])[..., :3]
            util.make_grid([x, rgb_yt, rgb_yp], 1, 3, str(epoch_dir / f'{i:03d}.jpg'))

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
