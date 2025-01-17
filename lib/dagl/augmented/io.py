
from pathlib import Path
from functools import partial
from easydict import EasyDict as edict

import torch as th

from .dagl import RR as DAGL
from ..utils import optional as _optional
from ..utils import select_sigma,remove_state_prefix
from ..utils.lin2conv import convert_lin2conv

# -- auto populate fields to extract config --
_fields = []
def optional_full(init,pydict,field,default):
    if not(field in _fields) and init:
        _fields.append(field)
    return _optional(pydict,field,default)

def load_model(cfg):

    # -- allows for all keys to be aggregated at init --
    init = _optional(cfg,'__init',False) # purposefully weird key
    optional = partial(optional_full,init)

    # -- unpack configs --
    arch_cfg = extract_arch_config(cfg,optional)
    search_cfg = extract_search_config(cfg,optional)
    io_cfg = extract_io_config(cfg,optional)
    mtype = optional(cfg,'model_type','augmented') # for base
    device = optional(cfg,'device','cuda:0')
    if init: return

    # -- init model --
    model = DAGL(arch_cfg,search_cfg)

    # -- load model --
    load_model_weights(model,io_cfg)

    # -- device --
    model = model.to(device)

    return model

def load_model_weights(model,cfg):
    if not(cfg.pretrained_load):
        return model
    else:
        print("Loading State: ",cfg.pretrained_path)
        state = th.load(cfg.pretrained_path,map_location=cfg.map_location)
        convert_lin2conv(state,cfg.pretrained_lin2conv)
        model.load_state_dict(state)
    return model

# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-
#     Configs Sets
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-

def extract_pairs(pairs,_cfg,optional):
    cfg = edict()
    for key,val in pairs.items():
        cfg[key] = optional(_cfg,key,val)
    return cfg

def extract_io_config(_cfg,optional):
    sigma = optional(_cfg,"sigma",0.)
    model_sigma = select_sigma(sigma)
    map_location = optional(_cfg,"map_location","cuda")
    base = Path("weights/checkpoints/DN_Gray/")
    pretrained_path = base / Path("%d/model/model_best.pt" % model_sigma)
    pretrained_lin2conv = False
    pairs = {"pretrained_load":True,
             "pretrained_path":str(pretrained_path),
             "pretrained_prefix":"",
             "map_location":map_location,
             "pretrained_lin2conv":pretrained_lin2conv}
    return extract_pairs(pairs,_cfg,optional)

def extract_search_config(_cfg,optional):
    pairs = {"attn_mode":"dnls_k",
             "k_s":100,"k_a":100,
             "ws":21,"ws_r":3,
             "ps":7,"pt":1,"wt":0,
             "stride0":4,"stride1":1,"bs":-1,
             "rbwd":True,"nbwd":1,"exact":False,
             "reflect_bounds":True,
             "refine_inds":False,
             "dilation":1,"return_inds":False,
             "use_pfc":False}
    return extract_pairs(pairs,_cfg,optional)

def extract_arch_config(_cfg,optional):
    pairs = {"scale":[1],"self_ensemble":False,
             "chop":False,"precision":"single",
             "cpu":False,"n_GPUs":1,"pre_train":".",
             "save_models":False,"model":"COLA","mode":"E",
             "print_model":False,"resume":0,"seed":1,
             "n_resblocks":16,"n_feats":64,"n_colors":1,
             "res_scale":1,"rgb_range":1.,"stages":6,
             "blocks":3,"act":"relu","sigma":0.,
             "arch_return_inds":False}
    return extract_pairs(pairs,_cfg,optional)

# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
#
#     Extracting Relevant Fields from Larger Dict
#
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

def extract_model_config(cfg):
    # -- auto populated fields --
    fields = _fields
    model_cfg = {}
    for field in fields:
        if field in cfg:
            model_cfg[field] = cfg[field]
    return model_cfg

# -- run to populate "_fields" --
load_model(edict({"__init":True}))

