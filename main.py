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

model = cnn01.get_model()
cnn01.train(model)