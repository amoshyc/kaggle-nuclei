import numpy as np
import torch
from torch.autograd import Variable

from nuclei import config
from nuclei.features.raw import *
from nuclei.loss.ae import *


def test_tag_loss_single_np():
    mask1 = np.float32([[1, 1, 0, 0], [1, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0,
                                                                   0]])
    mask2 = np.float32([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 0,
                                                                   1]])
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
    print('-' * 10)
    print(pred)
    print('tag_loss =', tag_loss_single_np(pred, coords))
    pred = np.float32([[2, 2, 2, 2], [2, 2, 2, 2], [3, 3, 3, 3], [3, 3, 3, 3]])
    print('-' * 10)
    print(pred)
    print('tag_loss =', tag_loss_single_np(pred, coords))
    pred = np.float32([[0, 0, 0, 0], [0, 0, 0, 0], [9, 9, 9, 9], [9, 9, 9, 9]])
    print('-' * 10)
    print(pred)
    print('tag_loss =', tag_loss_single_np(pred, coords))


def test_tag_loss_single():
    mask1 = np.float32([[1, 1, 0, 0], [1, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0,
                                                                   0]])
    mask2 = np.float32([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 0,
                                                                   1]])
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
    print('-' * 10)
    print(pred)
    print('tag_loss =', tag_loss_single(Variable(torch.from_numpy(pred).cuda()), coords))
    pred = np.float32([[2, 2, 2, 2], [2, 2, 2, 2], [3, 3, 3, 3], [3, 3, 3, 3]])
    print('-' * 10)
    print(pred)
    print('tag_loss =', tag_loss_single(Variable(torch.from_numpy(pred).cuda()), coords))
    pred = np.float32([[0, 0, 0, 0], [0, 0, 0, 0], [9, 9, 9, 9], [9, 9, 9, 9]])
    print('-' * 10)
    print(pred)
    print('tag_loss =', tag_loss_single(Variable(torch.from_numpy(pred).cuda()), coords))


def test_tag_loss():
    img_paths = list(config.TRAIN1.glob('*/images/*.png'))[:30]
    ms = read_masks(img_paths)
    cs = sample_points(ms, n_points=10)

    ys = Variable(torch.zeros(30, 448, 448).cuda(), requires_grad=False)
    print(tag_loss(ys, cs, pbar=True))



if __name__ == '__main__':
    test_tag_loss_single()