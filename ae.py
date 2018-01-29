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
from nuclei.models.ae import *


img_paths = list(config.TRAIN1.glob('*/images/*.png'))[:10]
xs, ss = read_imgs(img_paths)
ms = read_masks(img_paths)
ys = fuse_masks(ms)

xs = np.transpose(xs, (0, 3, 1, 2))
ys = np.transpose(ys, (0, 3, 1, 2))

model = Net().cuda()
xs_batch = torch.from_numpy(xs[:10]).cuda()
xs_batch = Variable(xs_batch)

pred = model(xs_batch)
print(pred.size())
