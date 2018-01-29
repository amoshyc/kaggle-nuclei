import numpy as np
from PIL import Image

from skimage.measure import label

def to_pil(arr):
    arr = np.squeeze(arr)
    arr = np.uint8(arr * 255)
    arr = np.clip(arr, 0, 255)
    return Image.fromarray(arr)


def rle(img):
    flat_img = img.ravel()
    


def cm_to_rle(cm, th=0.5):
    lbl_cm, num = label(cm > th, return_num=True)
    for i in range(1, num + 1):
        yield rle(lbl_cm == i)