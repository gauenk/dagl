


# -- torch --
import torch as th
from easydict import EasyDict as edict

# -- data --
import data_hub

# -- dagl --
import dagl
from dagl import optional



def test_fwd():

    # -- create dict --
    cfg = edict()
    cfg.dname = "set8"
    cfg.isize = "128_128"
    cfg.vid_name = "sunflower"
    cfg.nframes = 1
    cfg.frame_start = 0
    cfg.frame_end = 1
    cfg.sigma = 30
    cfg.device = "cuda:0"
    cfg.bw = True

    # -- load data --
    data,loaders = data_hub.sets.load(cfg)
    frame_start = optional(cfg,"frame_start",0)
    frame_end = optional(cfg,"frame_end",-1)
    indices = data_hub.filter_subseq(data.te,cfg.vid_name,frame_start,frame_end)
    index = indices[0]
    sample = data.te[index]
    noisy,clean = sample['noisy'],sample['clean']
    noisy,clean = noisy.to(cfg.device),clean.to(cfg.device)
    print("[%d] noisy.shape: " % index,noisy.shape)

    # -- load model pair --
    cfg.model_type = "original"
    model_cfg = dagl.original.extract_model_config(cfg)
    original = dagl.load_model(model_cfg)
    cfg.model_type = "augmented"
    model_cfg = dagl.extract_model_config(cfg)
    augmented = dagl.load_model(model_cfg)

    # -- denoise --
    deno_gt = original.model(noisy)
    deno_te = augmented(noisy)

    # -- compare --
    diff = th.sum((deno_gt - deno_te)**2).item()
    assert diff < 1e-8

def test_bwd():

    # -- create dict --
    cfg = edict()
    cfg.dname = "set8"
    cfg.isize = "128_128"
    cfg.vid_name = "sunflower"
    cfg.nframes = 1
    cfg.frame_start = 0
    cfg.frame_end = 1
    cfg.sigma = 30
    cfg.device = "cuda:0"
    cfg.bw = True

    # -- load data --
    data,loaders = data_hub.sets.load(cfg)
    frame_start = optional(cfg,"frame_start",0)
    frame_end = optional(cfg,"frame_end",-1)
    indices = data_hub.filter_subseq(data.te,cfg.vid_name,frame_start,frame_end)
    index = indices[0]
    sample = data.te[index]
    noisy,clean = sample['noisy']/255.,sample['clean']/255.
    noisy,clean = noisy.to(cfg.device),clean.to(cfg.device)
    print("[%d] noisy.shape: " % index,noisy.shape)

    # -- load model pair --
    cfg.model_type = "original"
    model_cfg = dagl.original.extract_model_config(cfg)
    original = dagl.load_model(model_cfg)
    cfg.model_type = "augmented"
    model_cfg = dagl.extract_model_config(cfg)
    augmented = dagl.load_model(model_cfg)

    # -- allow grads --
    noisy_gt = noisy.requires_grad_(True)
    noisy_te = noisy.requires_grad_(True)

    # -- denoise --
    deno_gt = original.model(noisy_gt)
    deno_te = augmented(noisy_te)

    # -- compare --
    diff = th.sum((deno_gt - deno_te)**2).item()
    assert diff < 1e-8

    # -- gradient --
    th.autograd.backward(deno_gt,clean)
    th.autograd.backward(deno_te,clean)

    # -- grads --
    grads_gt = noisy_gt.grad
    grads_te = noisy_te.grad

    # -- compare --
    diff = th.mean((grads_gt - grads_te)**2).item()
    assert diff < 1e-8
