from yacs.config import CfgNode as CN

_C = CN()

_C.MODEL = CN()
_C.MODEL.NAME = "ViT-B/32"

_C.DATASET = CN()
_C.DATASET.NAME = "flower102"
_C.DATASET.IMG_BATCH_SIZE = 16

_C.TRAIN = CN()
_C.TRAIN.PARAM_STD = 0.02
_C.TRAIN.N_CTX = 16
_C.TRAIN.CSC = False
_C.TRAIN.CTX_INIT = ""
_C.TRAIN.SEED = 1

_C.OPTIM = CN()
_C.OPTIM.NAME = "sgd"
_C.OPTIM.LR = 0.002
_C.OPTIM.MAX_EPOCH = 50
_C.OPTIM.LR_SCHEDULER = "cosine"
_C.OPTIM.WARMUP_EPOCH = 1
_C.OPTIM.WARMUP_TYPE = "constant"
_C.OPTIM.WARMUP_CONS_LR = 1e-5

def get_cfg_defaults():
    names = []
    with open(f"{_C.DATASET.NAME}_names.txt", 'r') as f:
        for line in f:
            names.append(line.replace('\'', '').strip())
    _C.DATASET.CLASSNAMES = names
    return _C.clone()