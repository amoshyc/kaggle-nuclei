import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
plt.style.use('seaborn')

import numpy as np
from sklearn.model_selection import train_test_split

from nuclei import config
from nuclei import util
from nuclei.unet2.feature import *
from nuclei.unet2.model import *
from nuclei.unet2.train import Trainer

# unet.test_model()
# test_masks_pixel_overlap(masks)
# test_fuse_masks()
# util.test_make_batch()
# util.test_rle_encode()

# ps = list(config.TRAIN1.glob('*/images/*.png'))
# xs = read_imgs(ps, (448, 448), pbar='Read Imgs')
# all_masks = read_masks(ps, pbar='Read Masks')
# ys = fuse_masks(all_masks, pbar='Fuse Masks')

# xs = np.transpose(xs, (0, 3, 1, 2))
# ys = np.transpose(ys, (0, 3, 1, 2))
# xt, xv, yt, yv, pt, pv = train_test_split(xs, ys, ps, test_size=0.2)
# np.savez('./data/unet2.npz', xt=xt, yt=yt, xv=xv, yv=yv, pt=pt, pv=pv)

data = np.load('./data/unet2.npz')
xt, yt, pt = data['xt'], data['yt'], data['pt']
xv, yv, pv = data['xv'], data['yv'], data['pv']

model = UNet()

ckpt_dir = util.new_ckpt_dir()
trainer = Trainer(model, ckpt_dir)
trainer.fit(xt, yt, xv, yv)
