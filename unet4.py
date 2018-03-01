import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
plt.style.use('seaborn')

import numpy as np
from sklearn.model_selection import train_test_split

from nuclei import config
from nuclei import util
from nuclei.unet4.feature import *
from nuclei.unet4.model import *
from nuclei.unet4.train import Trainer

# unet.test_model()
# test_fuse_masks()
# util.test_make_batch()
# util.test_rle_encode()

# ps = list(config.TRAIN1.glob('*/images/*.png'))
# xs = read_imgs(ps, (448, 448), pbar='Read Imgs')
# ms = read_masks(ps, pbar='Read Masks')
# ys = fuse_masks(ms, pbar='Fuse Masks')

# xs = np.transpose(xs, (0, 3, 1, 2))
# ys = np.transpose(ys, (0, 3, 1, 2))
# xt, xv, yt, yv = train_test_split(xs, ys, test_size=0.2)
# np.savez('./data/unet4.npz', xt=xt, yt=yt, xv=xv, yv=yv)

data = np.load('./data/unet4.npz')
xt, yt = data['xt'], data['yt']
xv, yv = data['xv'], data['yv']

model = UNet()

ckpt_dir = util.new_ckpt_dir()
trainer = Trainer(model, ckpt_dir)
trainer.fit(xt, yt, xv, yv)
