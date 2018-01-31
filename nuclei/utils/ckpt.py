from .. import config


def avaible_id():
    return len([p for p in config.CKPT.iterdir() if p.is_dir()])


def new_dir():
    path = config.CKPT / 'm{:05d}'.format(avaible_id())
    path.mkdir()
    return path