import numpy as np
from tqdm import tqdm
from PIL import Image
import skimage


def read_imgs(img_paths, size=(448, 448), pbar=True):
    if pbar:
        img_paths = tqdm(img_paths, desc='Reading Images')

    xs = np.zeros((len(img_paths), *size, 3), dtype=np.float32)
    for i, p in enumerate(img_paths):
        img = Image.open(p)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        img = img.resize(size)
        xs[i] = np.array(img) / 255

    return xs


def read_masks(img_paths, size=(448, 448), pbar=True):
    if pbar:
        img_paths = tqdm(img_paths, desc='Reading Masks')

    ys = np.zeros((len(img_paths), *size, 1), dtype=np.float32)
    for i, img_p in enumerate(img_paths):
        mask_paths = img_p.parent.parent.glob('masks/*.png')
        for mask_p in mask_paths:
            img = Image.open(mask_p)
            if img.mode != 'L':
                img = img.convert('L')
            img = img.resize(size)
            img = np.uint8(img)
            ys[i, img > 0, 0] = 1.0

    return ys