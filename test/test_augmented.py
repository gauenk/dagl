
import data_hub
import dagl
from easydict import EasyDict as edict
from dagl import optional

def test_fwd():

    # -- create dict --
    cfg = edict()
    cfg.dname = "set8"
    cfg.isize = "128_128"
    cfg.vid_name = "sunflower"
    cfg.nframes = 3
    cfg.frame_start = 0
    cfg.frame_end = 2
    cfg.sigma = 30
    cfg.device = "cuda:0"

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
    deno_gt = original(noisy)
    deno_te = augmented(noisy)

    # -- compare --
    diff = th.sum((deno_gt - deno_te)**2).item()
    assert diff < 1e-8
