import numpy as np
from tqdm import tqdm
from PIL import Image


def read_imgs(img_paths, size=(448, 448), pbar=True):
    if pbar:
        img_paths = tqdm(img_paths, desc='Reading Images')

    xs = np.zeros((len(img_paths), *size, 3), dtype=np.float32)
    sizes = []
    for i, p in enumerate(img_paths):
        img = Image.open(p)
        img = img.convert('RGB')
        sizes.append(img.size)
        img = img.resize(size)
        xs[i] = np.array(img) / 255

    return xs, sizes


def read_masks(img_paths, size=(448, 448), pbar=True):
    if pbar:
        img_paths = tqdm(img_paths, desc='Reading Masks')

    all_masks = []
    for path in img_paths:
        cur_masks = []
        cur_masks_paths = path.parent.parent.glob('masks/*.png')
        for mask_path in cur_masks_paths:
            img = Image.open(mask_path)
            img = img.convert('L')
            img = img.resize(size)
            cur_masks.append(np.uint8(img) / 255)
        all_masks.append(cur_masks)

    return all_masks


def fuse_masks(all_masks, size=(448, 448), pbar=True):
    if pbar:
        all_masks = tqdm(all_masks, desc='Fusing Masks')

    ys = np.zeros((len(all_masks), *size, 1), dtype=np.float32)
    for i, cur_masks in enumerate(all_masks):
        for mask in cur_masks:
            ys[i, mask > 0, 0] = 1.0
    
    return ys


def sample_points_from_mask(mask, n_points):
    rr, cc = np.nonzero(mask > 0)
    if len(rr) < n_points:
        indices = np.random.random_integers(0, len(rr) - 1, n_points)
    else:
        indices = np.arange(len(rr))
        np.random.shuffle(indices)
        indices = indices[:n_points]
    return rr[indices], cc[indices]


def sample_points(all_masks, n_points=5, pbar=True):
    if pbar:
        all_masks = tqdm(all_masks, desc='Sampling Points')

    res = []
    for cur_masks in all_masks:
        pts = np.zeros((len(cur_masks), 2, n_points), dtype=np.uint8)
        for i, mask in enumerate(cur_masks):
            rr, cc = sample_points_from_mask(mask, n_points)
            pts[i, 0, :] = rr
            pts[i, 1, :] = cc
        res.append(pts)
    
    return res

