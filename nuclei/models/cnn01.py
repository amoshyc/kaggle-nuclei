from keras.models import Model, load_model
from keras.layers import *
from keras.optimizers import *


def __conv(f, k, act='relu'):
    args = {
        'filters': c,
        'kernel_size': k,
        'activation': act,
        'padding': 'same',
        'kernel_initializer': 'he_normal' if act == 'relu' else 'random_normal'
    }
    return Conv2D(**args)


def model(size=(448, 448)):
    inp = Input(shape=size + (3, ))

    x = __conv(8, (5, 5))(inp)
    x = __conv(8, (5, 5))(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)

    x = __conv(16, (3, 3))(x)
    x = __conv(16, (3, 3))(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)

    x = __conv(16, (3, 3))(x)
    x = __conv(16, (3, 3))(x)
    x = BatchNormalization()(x)
    x = UpSampling2D((2, 2))(x)
    x = __conv(8, (5, 5))(x)
    x = __conv(8, (5, 5))(x)
    x = BatchNormalization()(x)
    out = UpSampling2D((2, 2))(x)

    return Model(inputs=inp, outputs=out)


def train(model, xt, yt, xv, yv):
    compile_arg = {
        'loss': 'binary_crossentropy',
        'optimizer': 'adam',
    }
    model.compile(**compile_arg)
    model.summary()

    fit_arg = {
        'x': xt, 'y': yt,
        'batch_size': 50,
        'epochs': 250,
        'shuffle': True,
        'validation_data': (xv, yv),
        # 'callbacks': [
        #     CSVLogger(str(log_path)),
        #     ModelCheckpoint(str(weights_path)),
        # ]
    }
    model.fit(**fit_arg)