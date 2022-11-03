
from pathlib import Path
from functools import partial
from easydict import EasyDict as edict

import torch as th

from . import dn_gray
from . import dn_real
from ..utils import optional as _optional
from ..utils import select_sigma,remove_state_prefix

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
    task = optional(cfg,"task","denoising_bw")
    mtype = optional(cfg,'model_type','original') # for base
    device = optional(cfg,'device','cuda:0')
    if init: return

    # -- init model --
    DAGL = load_model_task(task)
    model = DAGL(arch_cfg,arch_cfg.ckp)#,search_cfg)

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
        state = remove_state_prefix(cfg.pretrained_prefix,state)
        model.model.load_state_dict(state)
    return model

# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-
#     Configs for "io"
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-

def extract_pairs(pairs,_cfg,optional):
    cfg = edict()
    for key,val in pairs.items():
        cfg[key] = optional(_cfg,key,val)
    return cfg

def extract_io_config(_cfg,optional):
    sigma = optional(_cfg,"sigma",0.)
    map_location = optional(_cfg,"map_location","cuda")
    base = Path("weights/checkpoints/DN_Gray/")
    model_sigma = select_sigma(sigma)
    pretrained_path = base / Path("%d/model/model_best.pt" % model_sigma)
    pairs = {"pretrained_load":True,
             "pretrained_path":str(pretrained_path),
             "pretrained_prefix":"",
             "map_location":map_location}
    return extract_pairs(pairs,_cfg,optional)

def extract_search_config(_cfg,optional):
    pairs = {"attn_mode":"dnls_k",
             "k_s":200,"k_a":100,
             "ws":21,"ws_r":3,
             "ps":7,"pt":1,"wt":0,
             "stride0":4,"stride1":1,"bs":-1,
             "rbwd":True,"nbwd":1,"exact":False,
             "reflect_bounds":False,"refine_inds":[False,False,False],
             "dilation":1,"return_inds":True,}
    return extract_pairs(pairs,_cfg,optional)

def extract_arch_config(_cfg,optional):
    pairs = {"ckp":".",
             "scale":[1],"self_ensemble":False,
             "chop":False,"precision":"single",
             "cpu":False,"n_GPUs":1,"pre_train":".",
             "save_models":False,"model":"DAGL","mode":"E",
             "print_model":False,"resume":0,"seed":1,
             "n_resblocks":16,"n_feats":64,"n_colors":1,
             "res_scale":1,"rgb_range":1.,"stages":6,
             "blocks":3,"act":"relu","sigma":0.,
             "return_inds":False}
    return extract_pairs(pairs,_cfg,optional)

def load_model_task(task):
    if task in ["denoising_bw","denoising"]:
        return dn_gray.DAGL_model
    elif task == "denoising_real":
        return dn_real.DAGL_model
    else:
        raise ValueError(f"Uknown task [{task}]")

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
load_model({"__init":True})


#
#
# --- old code im keeping just in case --
#
#


# def load_bw_deno_network(data_sigma):

#     # -- params --
#     model_sigma = select_sigma(data_sigma)
#     args = dn_gray.default_options()
#     args.ensemble = False
#     checkpoint = edict()
#     checkpoint.dir = "."

#     # -- weights --
#     weights = "/home/gauenk/Documents/packages/dagl/weights/checkpoints/"
#     weights += ("DN_Gray/%d/model/model_best.pt" % model_sigma)
#     weights = Path(weights)

#     # -- model --
#     model = dn_gray.DAGL_model(args,checkpoint)
#     model.model.load_state_dict(th.load(weights,map_location='cuda'))
#     return model

# def load_real_deno_network():
#     # model_sigma = select_sigma(data_sigma)
#     weights = "/home/gauenk/Documents/packages/dagl/weights/checkpoints/"
#     weights += "DN_Real/rn/model/model_best.pt"
#     args = dn_real.get_options()
#     model = dn_real.DAGL(args)
#     model.load_state_dict(th.load(weights, map_location = 'cuda'))
#     return model

# def run_bw(model,image):
#     with th.no_grad():
#         deno = model(image)
#     return deno

# def run_real(model,image):
#     with th.no_grad():
#         deno = th.clamp(dn_real.forward_chop(x=image, nn_model=model,flg=0), 0., 1.)
#     return deno

