import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
plt.style.use('seaborn')

import numpy as np
from sklearn.model_selection import train_test_split

from nuclei import config
from nuclei import util
from nuclei.unet1.feature import *
from nuclei.unet1.model import *
from nuclei.unet1.predict import *
from nuclei.unet1.train import *


# test_model()
# test_masks_pixel_overlap(masks)
# test_fuse_masks()
# test_label()

# ps = list(config.TRAIN1.glob('*/images/*.png'))
# xs = read_imgs(ps, (448, 448), pbar='Read Imgs')
# all_masks = read_masks(ps, pbar='Read Masks')
# ys = fuse_masks(all_masks, pbar='Fuse Masks')

# xs = np.transpose(xs, (0, 3, 1, 2))
# ys = np.transpose(ys, (0, 3, 1, 2))
# xt, xv, yt, yv, pt, pv = train_test_split(xs, ys, ps, test_size=0.2)
# np.savez('./data/unet1.npz', xt=xt, yt=yt, xv=xv, yv=yv, pt=pt, pv=pv)

data = np.load('./data/unet1.npz')
# xt, yt, pt = data['xt'], data['yt'], data['pt']
xv, yv, pv = data['xv'], data['yv'], data['pv']

# ckpt_dir = util.new_ckpt_dir()
# model = UNet()
# trainer = Trainer(model, ckpt_dir)
# trainer.fit(xt, yt, xv, yv)

ckpt_dir = config.CKPT / 'm00001'
model = UNet()
model.load_state_dict(T.load(str(ckpt_dir / 'model.pth')))
pred = Predictor(model, ckpt_dir)
pred.fit(xv[:10], pv[:10])

