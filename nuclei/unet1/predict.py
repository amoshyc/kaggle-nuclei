import pathlib
import numpy as np
import pandas as pd
from tqdm import tqdm
from skimage import transform
from skimage import measure
from skimage import color
from skimage import io

import torch as T
from torch.autograd import Variable

from .. import util


class Predictor(object):
    def __init__(self, model, target_dir):
        super().__init__()
        self.model = model.cuda()
        self.batch_size = 10
        self.target_dir = pathlib.Path(target_dir)
        self.target_dir.mkdir(exist_ok=True)        

    def predict(self, xs, ps, plot=True):
        '''
            xs: ndarray with shape (N, C, W, H)
            ps: list of image paths corresponding to xs
        '''
        xs = T.from_numpy(xs)
        df = pd.DataFrame(columns=['ImageId', 'EncodedPixels'])

        self.model.eval()
        with tqdm(total=len(xs), ascii=True) as pbar:
            for xb, pb in util.make_batch(xs, ps, bs=self.batch_size):
                xb = Variable(xb.cuda(), requres_grad=False)
                yb = self.model(xb).cpu().data.numpy()
                for x, y, p in zip(xb, yb, pb):
                    # RLE
                    img_size = Image.open(p).size
                    fused = transform.resize(y[..., 0], img_size)
                    labeled, num = measure.label(fused > 0.5, return_num=True)
                    for i in range(num):
                        mask = np.uint8(labeled == num)
                        df = df.append({
                            'ImageId': p.stem, 
                            'EncodedPixels': util.rle_encode(mask),
                        }, ignore_index=True)
                    # PLOT
                    if plot:
                        vis = color.label2rgb(labeled, x)
                        io.imsave(str(self.target_dir / f'{p.stem}vis.jpg'), vis)
                pbar.update(self.batch_size)
        df.to_csv(str(self.target_dir / 'unet1.csv'))


def test_label():
    x = np.float32([
        [1, 1, 0, 0],
        [0, 0, 0, 1],
        [1, 0, 1, 0],
        [0, 0, 1, 1]
    ])
    labeled, num = measure.label(x > 0.5, return_num=True)
    print('num:', num)
    for i in range(num):
        mask = np.uint8(labeled == i)
        print('mask', i, ':')
        print(mask)
    vis = color.label2rgb(labeled, x)
    io.imsave('./vis.jpg', vis)