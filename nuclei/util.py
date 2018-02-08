from math import ceil

import numpy as np

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


def rle_encode(img, threshold=0.5):
    flat_img = img.ravel()
    flat_img = np.where(flat_img > threshold, 1, 0).astype(np.uint8)
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
    assert rle_encode(img, threshold=0.5) == '1 5 9 2 13 2'
    print('Pass')