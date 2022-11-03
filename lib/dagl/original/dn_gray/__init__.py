import torch
from .utility import checkpoint
# from . import option# import args
from .model.dagl import RR as DAGL
from .model import Model as DAGL_model
from easydict import EasyDict as edict


def get_options():
    return option.args

def default_options():
    args = edict()
    args.scale = [1]
    args.self_ensemble = False
    args.chop = True
    args.precision = "single"
    args.cpu = False
    args.n_GPUs = 1
    args.pre_train = "."
    args.save_models = False
    args.model = "dagl"
    args.resume = 0
    args.seed = 1
    args.n_resblocks = 16
    args.n_feats = 64
    args.n_colors = 1
    args.res_scale = 1
    args.rgb_range = 1.
    return args

