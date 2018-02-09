from math import ceil

import numpy as np
from PIL import Image

from . import config


def avaible_ckpt_id():
    return len([p for p in config.CKPT.iterdir() if p.is_dir()])


def new_ckpt_dir():
    path = config.CKPT / 'm{:05d}'.format(avaible_ckpt_id())
    path.mkdir()
    return path


def make_batch(*arrs, bs):
    assert len(arrs) > 0
    for arr in arrs:
        assert len(arr) == len(arrs[0])

    n_samples = len(arrs[0])
    for i in range(ceil(n_samples / bs)):
        s = i * bs
        t = min(s + bs, n_samples)
        yield tuple(arr[s:t] for arr in arrs)


def make_vis(arr, path):
    '''
        arr: list of float rgb ndarray (0 ~ 1)
    '''
    vis = np.uint8(np.hstack(arr) * 255)
    Image.fromarray(vis).save(str(path))


def make_grid(arr, nrows, ncols, path):
    '''
        arr: list of float rgb ndarray (0 ~ 1)
    '''
    rows = []
    for r in range(nrows):
        s = r * ncols
        t = min(s + ncols, len(arr))
        rows.append(np.hstack(arr[s:t]))
    vis = np.uint8(np.vstack(rows) * 255)
    Image.fromarray(vis).save(str(path))


def rle_encode(img):
    flat_img = np.uint8(img).ravel()
    flat_img = np.insert(flat_img, [0, len(flat_img)], [0, 0])

    starts = np.array((flat_img[:-1] == 0) & (flat_img[1:] == 1))
    ends = np.array((flat_img[:-1] == 1) & (flat_img[1:] == 0))
    starts_ix = np.where(starts)[0] + 1
    ends_ix = np.where(ends)[0] + 1
    lengths = ends_ix - starts_ix

    tokens = ['{} {}'.format(s, l) for s, l in zip(starts_ix, lengths)]
    return ' '.join(tokens).rstrip()


def test_make_batch():
    x = np.zeros((10, 2, 2))
    y = [1 for _ in range(10)]
    for xb, yb in make_batch(x, y, bs=3):
        assert len(xb) == len(yb)
    print('Pass')


def test_rle_encode():
    img = np.array([[1.0, 1.0, 0.8, 0.7], [1.0, 0.2, 0.3, 0.2],
                    [1.0, 1.0, 0.0, 0.0], [0.6, 0.6, 0.1, 0.1]])
    assert rle_encode(img > 0.5) == '1 5 9 2 13 2'
    print('Pass')