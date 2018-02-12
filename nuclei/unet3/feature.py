import pickle
import pathlib
from collections import defaultdict

import numpy as np
from PIL import Image
from tqdm import tqdm
from scipy.ndimage.measurements import center_of_mass
from skimage import draw
from skimage import color

from .. import config
from .. import util


def gaussian2d(w, h):
    sigma_x = w / 4
    sigma_y = h / 4
    x = np.arange(w)
    y = np.arange(h)
    gx = np.exp(-(x - ((w - 1) / 2))**2 / (2 * sigma_x**2))
    gy = np.exp(-(y - ((h - 1) / 2))**2 / (2 * sigma_y**2))
    g = np.outer(gy, gx)
    return g


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
    all_masks = []
    for i, path in enumerate(img_paths):
        mask_paths = path.parent.parent.glob('masks/*.png')
        all_masks.append([])
        for mask_path in mask_paths:
            img = Image.open(mask_path)
            img = img.convert('L')
            img = img.resize(size)
            img = np.uint8(img) > 0
            all_masks[-1].append(img)
    return all_masks


def fuse_masks(all_masks, size=(448, 448), pbar=None):
    '''
        ys.shape = (N, W, H, 2) where 
        ys[..., 0] = fused
        ys[..., 1] = marker
    '''
    if pbar:
        all_masks = tqdm(all_masks, desc=pbar)
    ys = np.zeros((len(all_masks), *size, 2), dtype=np.float32)
    for i, masks in enumerate(all_masks):
        for mask in masks:
            if mask.max() != 1.0:
                continue
            r, c = map(int, center_of_mass(mask))
            rr, cc = draw.circle(r, c, 4, shape=size)
            ys[i, mask, 0] = 1.0
            ys[i, rr, cc, 1] = 1.0
    return ys


def test_fuse_masks():
    n_samples = 10
    ps = list(config.TRAIN1.glob('*/images/*.png'))[:n_samples]
    xs = read_imgs(ps, pbar=False)
    ms = read_masks(ps, pbar=False)
    ys = fuse_masks(ms)

    for i in range(n_samples):
        x, y_s, y_m = xs[i], ys[i, ..., 0], ys[i, ..., 1]
        rgb_y_s = color.gray2rgb(y_s)
        rgb_y_m = color.gray2rgb(y_m)
        util.make_vis([x, rgb_y_s, rgb_y_m], f'./temp/vis{i:02d}.jpg')
