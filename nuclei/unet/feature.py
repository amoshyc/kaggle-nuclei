import pickle
import pathlib
from collections import defaultdict

import numpy as np
from PIL import Image
from tqdm import tqdm
from scipy.ndimage.morphology import binary_erosion


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
    if pbar:
        all_masks = tqdm(all_masks, desc=pbar)
    ys = np.zeros((len(all_masks), *size, 1), dtype=np.float32)
    for i, masks in enumerate(all_masks):
        for mask in masks:
            mask = binary_erosion(mask)
            ys[i, mask, 0] = 1.0
    return ys


def test_masks_pixel_overlap(all_masks):
    for masks in all_masks:
        cnt = np.sum(np.array(masks), axis=0)
        assert cnt.max() == 1
    print('Pass')


def test_fuse_masks():
    mask1 = np.zeros((10, 10))
    mask1[:4, :4] = 1
    mask2 = np.zeros((10, 10))
    mask2[4:8, :] = 1
    masks = np.float32((mask1 == 1) | (mask2 == 1))
    print('masks:')
    print(masks)
    res = fuse_masks([[mask1, mask2]], size=(10, 10), pbar=False)[0, ..., 0]
    print('fuse:')
    print(res)