import numpy as np
from PIL import Image, ImageOps
from keras.callbacks import Callback
import skimage
from skimage import color

from ..utils import convert


class Prediction(Callback):
    def __init__(self, xs, ys, ckpt_path):
        self.xs = xs
        self.ys = ys
        self.ckpt_path = ckpt_path / 'vis'
        self.ckpt_path.mkdir()

    def on_epoch_end(self, epoch, logs):
        epoch_dir = self.ckpt_path / f'{epoch:03d}'
        epoch_dir.mkdir()

        ps = self.model.predict(self.xs)
        for i, p in enumerate(ps):
            x = self.xs[i]
            y = color.gray2rgb(self.ys[i, ..., 0])
            p = np.concatenate([p * (255 / 255), p * (152 / 255), p * (0 / 255)], axis=-1)
            vis = convert.to_pil(np.hstack((x, y, p)))
            vis.save(str(epoch_dir / f'vis{i:03d}.jpg'))
