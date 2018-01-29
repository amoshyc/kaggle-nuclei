from keras import backend as K
from keras.models import Model, load_model
from keras.layers import *
from keras.losses import *
from keras.optimizers import *
from keras.callbacks import CSVLogger, ModelCheckpoint

from ..utils.ckpt import new_ckpt_path
from ..callbacks.plotloss import PlotLoss
from ..callbacks.pred import Prediction
from ..features.ae import read_imgs, read_masks, fuse_masks


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

    x = __conv(16, (3, 3))(inp)
    x = __conv(16, (3, 3))(x)
    x = __conv(16, (3, 3))(x)
    out1 = MaxPooling2D((2, 2))(x)

    x = __conv(32, (3, 3))(out1)
    x = __conv(32, (3, 3))(x)
    x = __conv(32, (3, 3))(x)
    out2 = MaxPooling2D((2, 2))(x)

    x = __conv(64, (1, 1))(out2)
    out1 = __conv(16, (3, 3))(out1)
    out2 = __conv(32, (3, 3))(out2)

    up2 = UpSampling2D(x)
    x = Concatenate()([up2, out2])
    x = __conv(32, (3, 3))(x)
    x = __conv(32, (3, 3))(x)
    x = __conv(32, (3, 3))(x)

    up1 = UpSampling2D((2, 2))(x)
    x = Concatenate()([up1, out1])
    x = __conv(16, (3, 3))(x)
    x = __conv(16, (3, 3))(x)
    x = __conv(16, (3, 3))(x)

    out = __conv(2, (3, 3), act='linear')(x)

    return Model(inputs=inp, outputs=out)


def ae_loss(y_true, y_pred):
    global masks

    losses = []
    for i in range(len(y_true)):
        mask = masks[i]

        tag = y_pred[..., 0]
        heatmap = K.sigmoid(y_pred[..., 0])
        loss_mse = mean_squared_error(y_true, heatmap)
        
        res = list([K.mean(tag[det > 0]) for det in mask])
        loss_person = K.mean([K.square(tag[det > 0] - re) for det, re in zip(mask, res)])
        loss_people = K.mean(K.exp((-1/2) * (re1 - re2)) for re1 in res for re2 in res)
        loss_group = loss_person + loss_people

        loss = loss_mes + 0.1 * loss_group
        losses.append(loss)

    return np.array(losses).mean()


def train(model):
    global masks

    img_paths = list(config.TRAIN1.glob('*/images/*.png'))[:100]
    xs = read_imgs(img_paths)
    masks = read_masks(img_paths)
    ys = fuse_masks(masks)

    xt, xv, yt, yv = train_test_split(xs, ys, test_size=0.2)

    compile_arg = {
        'loss': ae_loss,
        'optimizer': 'adam',
    }
    model.compile(**compile_arg)
    model.summary()

    # ckpt_path = new_ckpt_path()
    # log_path = ckpt_path / 'log.csv'
    # ws_path = ckpt_path / 'ws' / '{epoch:03d}_{val_loss:.3f}.h5'
    # ws_path.parent.mkdir()
    # print('CKPT:', ckpt_path)

    fit_arg = {
        'x': xt, 'y': yt,
        'batch_size': 50,
        'epochs': 250,
        'shuffle': True,
        'validation_data': (xv, yv),
        'callbacks': [
            # CSVLogger(str(log_path)),
            # ModelCheckpoint(str(ws_path)),
            # PlotLoss(ckpt_path),
            # Prediction(xv[:20], yv[:20], ckpt_path)
        ]
    }
    model.fit(**fit_arg)