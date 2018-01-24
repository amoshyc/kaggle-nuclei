import numpy as np
from PIL import Image

def to_pil(arr):
    arr = np.squeeze(arr)
    arr = np.uint8(arr * 255)
    arr = np.clip(arr, 0, 255)
    return Image.fromarray(arr)