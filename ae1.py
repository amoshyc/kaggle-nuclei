from pprint import pprint

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
plt.style.use('seaborn')

import numpy as np
from skimage import draw
from sklearn.model_selection import train_test_split

from nuclei import config
from nuclei.features.raw import *
from nuclei.models import ae1


img_paths = list(config.TRAIN1.glob('*/images/*.png'))
xs, ss = read_imgs(img_paths)
ms = read_masks(img_paths)
bs = to_border(ms)
ys = fuse_masks(ms)
cs = sample_points(bs, n_points=30)


xs = np.transpose(xs, (0, 3, 1, 2))
ys = np.transpose(ys, (0, 3, 1, 2))
xt, xv, yt, yv, mt, mv, ct, cv = train_test_split(xs, ys, ms, cs, test_size=0.2)

model = ae1.Net()
ae1.train(model, xt, yt, mt, ct, xv, yv, mv, cv)
