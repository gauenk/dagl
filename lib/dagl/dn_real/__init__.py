
import torch
from . import option# import args
from .model.dagl import RR as DAGL
from easydict import EasyDict as edict
from .test import forward_chop

def get_options():
    return option.args
