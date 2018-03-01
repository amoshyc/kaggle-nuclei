import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
plt.style.use('seaborn')

import numpy as np
from sklearn.model_selection import train_test_split

from nuclei import config
from nuclei import util
from nuclei.ae1.feature import *
from nuclei.ae1.model import *
from nuclei.ae1.train import Trainer

# test_model()
# test_fuse_masks()

ps = list(config.TRAIN1.glob('*/images/*.png'))[:200]
xs = read_imgs(ps, (448, 448), pbar='Read Imgs')
ms = read_masks(ps, pbar='Read Masks')
ys = fuse_masks(ms, pbar='Fuse Masks')

xs = np.transpose(xs, (0, 3, 1, 2))
ys = np.transpose(ys, (0, 3, 1, 2))
xt, xv, yt, yv, mt, mv = train_test_split(xs, ys, ms, test_size=0.2)

np.savez('./data/ae1.npz', xt=xt, yt=yt, mt=mt, xv=xv, yv=yv, mv=mv)
data = np.load('./data/ae1.npz')
xt, yt, mt = data['xt'], data['yt'], data['mt']
xv, yv, mv = data['xv'], data['yv'], data['mv']

model = AE()

ckpt_dir = util.new_ckpt_dir()
trainer = Trainer(model, ckpt_dir)
trainer.fit(xt, yt, mt, xv, yv, mv)
