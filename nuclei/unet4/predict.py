import pathlib
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image
from skimage import transform
from skimage import measure
from skimage import color
from skimage import morphology
from scipy import ndimage as ndi

import torch as T
from torch.autograd import Variable

from .. import util


def colorize(yp):
    segment = yp[..., 0] > 0.5
    markers = yp[..., 1] > 0.5
    labeled = morphology.watershed(segment, morphology.label(markers), mask=segment)
    vis = color.label2rgb(labeled, None, bg_label=0)
    return vis