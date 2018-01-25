from .. import config


def avaible_ckpt_id():
    return len([p for p in config.CKPT.iterdir() if p.is_dir()])


def new_ckpt_path():
    path = config.CKPT / 'm{:05d}'.format(avaible_ckpt_id())
    path.mkdir()
    return path