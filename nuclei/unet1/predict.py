import pathlib
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image
from skimage import transform
from skimage import measure
from skimage import color
from skimage import io

import torch as T
from torch.autograd import Variable

from .. import util


class Predictor(object):
    def __init__(self, model, ckpt_dir):
        super().__init__()
        self.model = model.cuda()
        self.batch_size = 5
        self.target_dir = ckpt_dir / 'pred'
        self.target_dir.mkdir(exist_ok=True)

    def fit(self, xs, ps, plot=True):
        '''
            xs: ndarray with shape (N, C, W, H)
            ps: list of image paths corresponding to xs
        '''
        xs = T.from_numpy(xs)
        df = pd.DataFrame(columns=['ImageId', 'EncodedPixels'])

        self.model.eval()
        with tqdm(total=len(xs), ascii=True) as pbar:
            for xb, pb in util.make_batch(xs, ps, bs=self.batch_size):
                xb_var = Variable(xb.cuda(), requires_grad=False)
                yb_var = self.model(xb_var)
                xb = np.transpose(xb_var.cpu().data.numpy(), [0, 2, 3, 1])
                yb = np.transpose(yb_var.cpu().data.numpy(), [0, 2, 3, 1])

                for x, y, p in zip(xb, yb, pb):
                    # PLOT
                    if plot:
                        labeled, num = measure.label(y[..., 0] > 0.5, return_num=True)
                        rgb = color.label2rgb(labeled, x, bg_label=0, alpha=0.2)
                        util.make_vis([x, rgb], (self.target_dir / f'{p.stem}vis.jpg'))

                    # RLE
                    img_size = Image.open(p).size
                    y = transform.resize(y[..., 0], img_size, mode='reflect')
                    labeled, num = measure.label(y > 0.5, return_num=True)
                    for i in range(1, num+1):
                        mask = np.uint8(labeled == i)
                        df = df.append({
                            'ImageId': p.stem, 
                            'EncodedPixels': util.rle_encode(mask),
                        }, ignore_index=True)
                pbar.update(self.batch_size)
        df.to_csv(str(self.target_dir / 'pred.csv'), index=False)


def test_label():
    x = np.zeros((100, 100), dtype=np.float32)
    x[20:40, 30:70] = 1.0
    x[50:60, 60:80] = 1.0
    labeled, num = measure.label(x > 0.5, return_num=True)
    assert num == 2
    vis = color.label2rgb(labeled, x, alpha=0.5, bg_label=0)
    Image.fromarray(np.uint8(vis * 255)).save('./vis.jpg')