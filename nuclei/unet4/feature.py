import pickle
import pathlib
from collections import defaultdict

import matplotlib.pyplot as plt

import numpy as np
from PIL import Image
from tqdm import tqdm
from scipy.ndimage.measurements import center_of_mass
from skimage import draw
from skimage import color

from .. import config
from .. import util


def read_imgs(img_paths, size=(448, 448), pbar=None):
    if pbar:
        img_paths = tqdm(img_paths, desc=pbar)
    xs = np.zeros((len(img_paths), *size, 3), dtype=np.float32)
    for i, path in enumerate(img_paths):
        img = Image.open(path)
        img = img.convert('RGB')
        img = img.resize(size)
        xs[i] = np.float32(img) / 255
    return xs


def read_masks(img_paths, size=(448, 448), pbar=None):
    if pbar:
        img_paths = tqdm(img_paths, desc=pbar)
    ms = np.zeros((len(img_paths), *size), dtype=np.uint8)
    for i, path in enumerate(img_paths):
        cnt_masks = 0
        mask_paths = path.parent.parent.glob('masks/*.png')
        for mask_path in mask_paths:
            img = Image.open(mask_path)
            img = img.convert('L')
            img = img.resize(size)
            img = np.uint8(img) > 0
            if np.any(img):
                cnt_masks += 1
                ms[i, img] = cnt_masks
    return ms


def fuse_masks(ms, size=(448, 448), pbar=None):
    if pbar:
        ms = tqdm(ms, desc=pbar)
    ys = np.zeros((len(ms), *size, 1), dtype=np.float32)
    for i, m in enumerate(ms):
        centers = center_of_mass((m > 0), m, np.arange(1, m.max() + 1))
        for id_ in range(1, m.max() + 1):
            r, c = centers[id_ - 1]
            rr, cc = np.nonzero(m == id_)
            angles = np.arctan2(-(rr - r), (cc - c))
            angles = angles + np.pi
            angles = angles / angles.max()
            ys[i, rr, cc, 0] = angles
    return ys


def test_fuse_masks():
    n_samples = 10
    ps = list(config.TRAIN1.glob('*/images/*.png'))[:n_samples]
    xs = read_imgs(ps, pbar=None)
    ms = read_masks(ps, pbar=None)
    ys = fuse_masks(ms)
    cm = plt.cm.spectral

    for i in range(n_samples):
        x = xs[i]
        y = cm(ys[i, ..., 0])[..., :3]
        util.make_vis([x, y], f'./temp/vis{i:02d}.jpg')
