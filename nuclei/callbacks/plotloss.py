import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.callbacks import Callback


class PlotLoss(Callback):
    def __init__(self, ckpt_path):
        self.ckpt_path = ckpt_path

    def on_epoch_end(self, epoch, logs):
        df = pd.read_csv(str(self.ckpt_path / 'log.csv'))
        keys = ['loss', 'val_loss']

        fig, ax = plt.subplots(dpi=150)
        df[keys].plot(kind='line', ax=ax)
        ax.set_xlabel('epoch')
        ax.set_ylabel('loss')
        fig.tight_layout()
        fig.savefig(str(self.ckpt_path / 'loss.png'))
        plt.close()
