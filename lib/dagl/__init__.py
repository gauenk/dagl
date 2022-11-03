# -- api --
from . import dn_gray
from . import dn_real
from . import utils


# -- function imports --
import os
import numpy as np
import torch as th
from pathlib import Path
from easydict import EasyDict as edict

def select_sigma(sigma):
    sigmas = np.array([15, 25, 50])
    msigma = np.argmin((sigmas - sigma)**2)
    return sigmas[msigma]

def load_bw_deno_network(data_sigma):

    # -- params --
    model_sigma = select_sigma(data_sigma)
    args = dn_gray.default_options()
    args.ensemble = False
    checkpoint = edict()
    checkpoint.dir = "."

    # -- weights --
    weights = "/home/gauenk/Documents/packages/dagl/weights/checkpoints/"
    weights += ("DN_Gray/%d/model/model_best.pt" % model_sigma)
    weights = Path(weights)

    # -- model --
    model = dn_gray.DAGL_model(args,checkpoint)
    model.model.load_state_dict(th.load(weights,map_location='cuda'))
    return model

def load_real_deno_network():
    # model_sigma = select_sigma(data_sigma)
    weights = "/home/gauenk/Documents/packages/dagl/weights/checkpoints/"
    weights += "DN_Real/rn/model/model_best.pt"
    args = dn_real.get_options()
    model = dn_real.DAGL(args)
    model.load_state_dict(th.load(weights, map_location = 'cuda'))
    return model

def run_bw(model,image):
    with th.no_grad():
        deno = model(image)
    return deno

def run_real(model,image):
    with th.no_grad():
        deno = th.clamp(dn_real.forward_chop(x=image, nn_model=model,flg=0), 0., 1.)
    return deno

