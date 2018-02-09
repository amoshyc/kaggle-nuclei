import pathlib
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image
from skimage import transform
from skimage import measure
from skimage import color

import torch as T
from torch.autograd import Variable

from .. import util


def label(yp):
    segment = yp[..., 0] > 0.5
    contour = yp[..., 1] > 0.5
    fused = (segment & (~contour))
    return measure.label(fused, return_num=True)


def colorize(yp):
    labeled, num = label(yp)
    vis = color.label2rgb(labeled, None, bg_label=0)
    return vis


def extract_masks(yp):
    labeled, num = label(yp)
    for i in range(1, num + 1):
        mask = (labeled == i)
        yield mask
