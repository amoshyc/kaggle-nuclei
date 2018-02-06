from math import ceil
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


def test_make_batch():
    import numpy as np
    x = np.zeros((10, 2, 2))
    y = [1 for _ in range(10)]
    for xb, yb in make_batch(x, y, bs=3):
        assert len(xb) == len(yb)
    print('Pass')