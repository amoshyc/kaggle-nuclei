from pprint import pprint

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
plt.style.use('seaborn')

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))

import numpy as np
from sklearn.model_selection import train_test_split

from nuclei import config
from nuclei.utils import convert
from nuclei.features import clahe
from nuclei.features import raw
from nuclei.models import cnn01

# img_paths = list(config.TRAIN1.glob('*/images/*.png'))
# xs = raw.read_imgs(img_paths)
# ys = raw.read_masks(img_paths)
# # np.savez('./data/data.npz', xs=xs, ys=ys)

# # data = np.load('./data/data.npz')
# # xs = data['xs']
# # ys = data['ys']

# xt, xv, yt, yv = train_test_split(xs, ys, test_size=0.2)

# model = cnn01.model()

# cnn01.train(model, xt, yt, xv, yv)

x = np.zeros((10, 10), dtype=np.float32)
x[2:5, 1:3] = 1.0
print(convert.cm_to_rle(x))