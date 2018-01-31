import numpy as np
from PIL import Image


def colorize(img, r=0, g=0, b=0, cmap=None):
    if cmap == 'orange':
        r, g, b = 255, 152, 0
    if cmap == 'white':
        r, g, b = 255, 255, 255

    return np.concatenate(
        [img * (r / 255), img * (g / 255), img * (b / 255)], axis=-1)


def to_pil_img(np_img):
    return Image.fromarray(np.uint8(np_img * 255))