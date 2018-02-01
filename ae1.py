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


def tag_loss(tag_map, mask_coords):
    n_masks = len(mask_coords)
    res = np.zeros((n_masks), dtype=np.float32)
    loss1 = np.zeros((n_masks), dtype=np.float32)
    loss2 = np.zeros((n_masks * n_masks), dtype=np.float32)
    for i, (rr, cc) in enumerate(mask_coords):
        res[i] = np.mean(tag_map[rr, cc])
        loss1[i] = np.mean((tag_map[rr, cc] - res[i]) ** 2)
    for i in range(n_masks):
        for j in range(n_masks):
            loss2[i * n_masks + j] = (res[i] - res[j]) ** 2
    loss1 = np.mean(loss1)
    loss2 = np.mean(np.exp(-1/2 * loss2))
    print('-' * 50)
    print(tag_map)
    print('res:', res)
    print('loss1 =', loss1)
    print('loss2 =', loss2)
    return loss1 + loss2


def test_sample_points_from_mask():
    mask = np.zeros((10, 10), dtype=np.uint8)
    rr, cc = draw.circle(5, 5, 4)
    mask[rr, cc] = 1
    print(mask)
    rr, cc = sample_points_from_mask(mask, 10)
    mask[rr, cc] = 2
    print(mask)


def test_tag_loss():
    mask1 = np.float32([[1, 1, 0, 0], [1, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
    mask2 = np.float32([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 0, 1]])
    print('mask1:')
    print(mask1)
    print('mask2:')
    print(mask2)

    masks = [mask1, mask2]
    all_masks = [masks]
    coords = sample_points(all_masks, n_points=2, pbar=False)[0]

    img = np.zeros((4, 4))
    rr1, cc1 = coords[0]
    rr2, cc2 = coords[1]
    img[rr1, cc1] = 1
    img[rr2, cc2] = 2
    print('sample:')
    print(img)

    pred = np.float32([[2, 2, 2, 3], [2, 3, 2, 3], [3, 3, 3, 3], [3, 3, 3, 4]])
    print('tag_loss =', tag_loss(pred, coords))
    pred = np.float32([[2, 2, 2, 2], [2, 2, 2, 2], [3, 3, 3, 3], [3, 3, 3, 3]])
    print('tag_loss =', tag_loss(pred, coords))
    pred = np.float32([[0, 0, 0, 0], [0, 0, 0, 0], [9, 9, 9, 9], [9, 9, 9, 9]])
    print('tag_loss =', tag_loss(pred, coords))

test_tag_loss()

# img_paths = list(config.TRAIN1.glob('*/images/*.png'))[:200]
# xs, ss = read_imgs(img_paths)
# ms = read_masks(img_paths)
# ys = fuse_masks(ms)

# xs = np.transpose(xs, (0, 3, 1, 2))
# ys = np.transpose(ys, (0, 3, 1, 2))
# xt, xv, yt, yv, mt, mv = train_test_split(xs, ys, ms, test_size=0.2)

# model = ae1.Net()
# ae1.train(model, xt, yt, mt, xv, yv, mv)
