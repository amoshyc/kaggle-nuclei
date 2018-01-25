from keras.models import Model, load_model
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import CSVLogger, ModelCheckpoint

from ..utils.ckpt import new_ckpt_path
from ..callbacks.plotloss import PlotLoss
from ..callbacks.pred import Prediction


def __conv(c, k, act='relu'):
    args = {
        'filters': c,
        'kernel_size': k,
        'activation': act,
        'padding': 'same',
        'kernel_initializer': 'random_normal'
    }
    return Conv2D(**args)


def model(size=(448, 448)):
    inp = Input(shape=size + (3, ))

    x = __conv(8, (5, 5))(inp)
    x = __conv(8, (5, 5))(x)
    x = MaxPooling2D((2, 2))(x)

    x = __conv(16, (3, 3))(x)
    x = __conv(16, (3, 3))(x)
    x = MaxPooling2D((2, 2))(x)

    x = __conv(32, (3, 3))(x)

    x = __conv(16, (3, 3))(x)
    x = __conv(16, (3, 3))(x)
    x = UpSampling2D((2, 2))(x)
    x = __conv(8, (5, 5))(x)
    x = __conv(8, (5, 5))(x)
    x = UpSampling2D((2, 2))(x)

    out = __conv(1, (3, 3), act='sigmoid')(x)

    return Model(inputs=inp, outputs=out)


def train(model, xt, yt, xv, yv):
    compile_arg = {
        'loss': 'binary_crossentropy',
        'optimizer': 'adam',
    }
    model.compile(**compile_arg)
    model.summary()

    ckpt_path = new_ckpt_path()
    log_path = ckpt_path / 'log.csv'
    ws_path = ckpt_path / 'ws' / '{epoch:03d}_{val_loss:.3f}.h5'
    ws_path.parent.mkdir()
    print('CKPT:', ckpt_path)

    fit_arg = {
        'x': xt, 'y': yt,
        'batch_size': 50,
        'epochs': 250,
        'shuffle': True,
        'validation_data': (xv, yv),
        'callbacks': [
            CSVLogger(str(log_path)),
            ModelCheckpoint(str(ws_path)),
            PlotLoss(ckpt_path),
            Prediction(xv[:20], yv[:20], ckpt_path)
        ]
    }
    model.fit(**fit_arg)