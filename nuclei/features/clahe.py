from tqdm import tqdm
from skimage import exposure

from . import raw


def read_imgs(img_paths, size=(448, 448), pbar=True):
    xs = raw.read_imgs(img_paths, size=size, pbar=pbar)

    if pbar:
        iters = tqdm(xs, desc='CLAHEing')
    for i, x in enumerate(iters):
        xs[i] = exposure.equalize_adapthist(x)

    return xs


def read_masks(img_paths, size=(448, 448), pbar=True):
    return raw.read_masks(img_paths, size=size, pbar=pbar)