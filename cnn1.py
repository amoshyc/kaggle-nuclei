from pprint import pprint

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
plt.style.use('seaborn')

import torch
from torch.autograd import Variable

import numpy as np
from sklearn.model_selection import train_test_split

from nuclei import config
from nuclei.features.raw import *
from nuclei.models import cnn1


img_paths = list(config.TRAIN1.glob('*/images/*.png'))
xs, ss = read_imgs(img_paths)
ms = read_masks(img_paths)
ys = fuse_masks(ms)

xs = np.transpose(xs, (0, 3, 1, 2))
ys = np.transpose(ys, (0, 3, 1, 2))
xt, xv, yt, yv = train_test_split(xs, ys, test_size=0.2)

model = cnn1.Net()
cnn1.train(model, xt, yt, xv, yv)
