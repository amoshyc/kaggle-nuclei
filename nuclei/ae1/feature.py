import pickle
import pathlib

import numpy as np
from PIL import Image
from tqdm import tqdm
from skimage import draw
from skimage import color
from skimage import measure
from scipy.ndimage.measurements import center_of_mass

import matplotlib.pyplot as plt


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
                ms[i, img] = cnt_masks
                cnt_masks += 1
    return ms


def fuse_masks(masks, size=(448, 448), pbar=None):
    if pbar:
        masks = tqdm(masks, desc=pbar)
    ys = np.zeros((len(masks), *size, 1), dtype=np.float32)
    for i, m in enumerate(masks):
        ys[i, m > 0, 0] = 1.0
    return ys


def test_fuse_masks():
    pass
