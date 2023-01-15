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
    # cfg.isize = "512_512"
    # cfg.nframes = 3
    # cfg.frame_start = 0
    # cfg.frame_end = cfg.frame_start+cfg.nframes-1
    cfg.bw = True
    cfg.return_inds = True
    cfg.ca_fwd = 'dnls_k'
    cfg.use_pfc = False

    # -- archs --
    cfg.arch_return_inds = True
    cfg.burn_in = True

    # -- processing --
    cfg.spatial_crop_size = "none"
    cfg.spatial_crop_overlap = 0.#0.1
    cfg.temporal_crop_size = 7#cfg.nframes
    cfg.temporal_crop_overlap = 0/5.#4/5. # 3 of 5 frames
    cfg.attn_mode = "dnls_k"
    cfg.use_chop = "false"

    # -- get config --
    cfg.k_s = 100
    cfg.k_a = 100
    cfg.ws = 27
    cfg.wt = 3
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
    # del cfg['isize']
    return cfg

def load_results(cfg,dnames,vid_names,sigmas):

    # -- get cache --
    home = Path(__file__).parents[0] / "../../../"
    cache_dir = str(home / ".cache_io")
    cache_name = "test_rgb_net_cvpr23"
    cache = cache_io.ExpCache(cache_dir,cache_name)

    # -- update with default --
    cfg = merge_with_base(cfg)
    detailed_cfg(cfg)
    use_train = ["true"]
    flow = ["true"]
    model_type = ['augmented']
    aug_test = ["true","false"]
    aug_refine_inds = ["true"]
    model_type = ['augmented']
    ws_r = [1,3]
    refine_inds = [("f-"*12)[:-1],("f-t-t-t-"*3)[:-1],
                   ("f-"+"t-"*11)[:-1],("f-t-t-t-t-t-"*2)[:-1],('t-'*12)[:-1]]
    exp_lists = {"dname":dnames,"vid_name":vid_names,"sigma":sigmas,
                 "flow":flow,"use_train":use_train,"ws_r":ws_r,
                 "model_type":model_type,"refine_inds":refine_inds,
                 "aug_refine_inds":aug_refine_inds,"aug_test":aug_test}
    exps = cache_io.mesh_pydicts(exp_lists) # create mesh
    cache_io.append_configs(exps,cfg)
    pp.pprint(exps[-1])

    # -- read --
    root = Path("./.icml23")
    if not root.exists():
        root.mkdir()
    pickle_store = str(root / "dagl_set8_results.pkl")
    records = cache.load_flat_records(exps,save_agg=pickle_store,clear=False)

    # -- standardize col names --
    # records = records.rename(columns={"sb":"batch_size"})

    return records
