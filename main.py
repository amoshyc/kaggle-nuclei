from pprint import pprint

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
plt.style.use('seaborn')

import numpy as np
from sklearn.model_selection import train_test_split

from nuclei import config
from nuclei.utils import convert
from nuclei.features import raw
from nuclei.models import cnn01

# img_paths = list(config.TRAIN1.glob('*/images/*.png'))[:20]
# xs = raw.read_imgs(img_paths)
# ys = raw.read_masks(img_paths)

# xt, xv, yt, yv = train_test_split(xs, ys, test_size=0.2)

from chainer.datasets import mnist

train, test = mnist.get_mnist()
print(train[0][0].shape)

# model = cnn01.CNN01()

# cnn01.train(model,train, test, None, None)

