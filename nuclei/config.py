import pathlib

DATASET = pathlib.Path('./data/').resolve()
TRAIN1 = DATASET / 'stage1_train'
TEST1 = DATASET / 'stage1_test'

TEMP = pathlib.Path('./temp/').resolve()
CKPT = pathlib.Path('./ckpt/').resolve()