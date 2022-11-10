
from easydict import EasyDict as edict
from .common import *

def default():
    cfg = edict()
    cfg.dname = "set8"
    cfg.bw = True
    cfg.device = "cuda:0"
    cfg.dset = "te"
    cfg.saved_dir = "./output/saved_dir/"
    cfg.seed = 123
    return cfg
