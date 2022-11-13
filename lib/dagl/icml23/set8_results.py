"""

I am an API to write the paper for CVPR

"""

# -- misc --
import os,math,tqdm
import pprint
pp = pprint.PrettyPrinter(indent=4)
import copy
dcopy = copy.deepcopy

# -- linalg --
import numpy as np
import torch as th
from einops import rearrange,repeat

# -- data mngmnt --
from pathlib import Path
from easydict import EasyDict as edict

# -- data --
import data_hub

# -- caching results --
import cache_io

# -- network --
import dagl

def detailed_cfg(cfg):

    # -- data config --
    # cfg.isize = "128_128"
    cfg.isize = "512_512"
    cfg.bw = True
    cfg.nframes = 3
    cfg.frame_start = 0
    cfg.frame_end = cfg.frame_start+cfg.nframes-1
    cfg.return_inds = True
    cfg.ca_fwd = 'dnls_k'

    # -- processing --
    cfg.spatial_crop_size = 512
    cfg.spatial_crop_overlap = 0.#0.1
    cfg.temporal_crop_size = 3#cfg.nframes
    cfg.temporal_crop_overlap = 0/5.#4/5. # 3 of 5 frames
    cfg.attn_mode = "dnls_k"
    cfg.use_chop = "false"

    # -- get config --
    cfg.k_s = 75
    cfg.k_a = 25
    cfg.ws = 27
    cfg.wt = 3
    cfg.ws_r = 1
    # cfg.bs = 48*1024
    # return cfg

def merge_with_base(cfg):
    # -- [first] merge with base --
    cfg_og = dcopy(cfg)
    cfg_l = [cfg]
    cfg_base = dagl.configs.test_rgb_vid.default()
    cache_io.append_configs(cfg_l,cfg_base)
    cfg = cfg_l[0]

    # -- overwrite with input values --
    for key in cfg_og:
        cfg[key] = cfg_og[key]

    # -- remove extra keys --
    del cfg['isize']
    return cfg

# def load_proposed(cfg,use_train="true",flow="true"):
#     use_chop = "false"
#     ca_fwd = "dnls_k"
#     sb = 256
#     return load_results(ca_fwd,use_train,use_chop,flow,sb,cfg)

# def load_original(cfg,use_chop="false"):
#     flow = "false"
#     use_train = "false"
#     ca_fwd = "default"
#     sb = 1
#     return load_results(ca_fwd,use_train,use_chop,flow,sb,cfg)

def load_results(cfg,vid_names):

    # -- get cache --
    home = Path(__file__).parents[0] / "../../../"
    cache_dir = str(home / ".cache_io")
    cache_name = "test_rgb_net_cvpr23"
    cache = cache_io.ExpCache(cache_dir,cache_name)

    # -- update with default --
    cfg = merge_with_base(cfg)
    detailed_cfg(cfg)
    flow,isizes = ["true"],["none"]
    use_train = ["true"]
    refine_inds = [("f-"*12)[:-1],("f-t-t-t-"*3)[:-1],
                   ("f-"+"t-"*11)[:-1],("f-t-t-t-t-t-"*2)[:-1]]
    model_type = ['augmented']
    exp_lists = {"refine_inds":refine_inds,"use_train":use_train,
                 "use_chop":["false"],
                 "model_type":model_type,"vid_name":vid_names}
    exps = cache_io.mesh_pydicts(exp_lists) # create mesh
    cache_io.append_configs(exps,cfg)
    pp.pprint(exps[0])

    # -- read --
    root = Path("./.cvpr23")
    if not root.exists():
        root.mkdir()
    pickle_store = str(root / "dagl_set8_results.pkl")
    records = cache.load_flat_records(exps,save_agg=pickle_store,clear=True)

    # -- standardize col names --
    records = records.rename(columns={"sb":"batch_size"})
    return records
