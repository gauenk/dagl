
# -- misc --
import os,math,tqdm
import pprint
pp = pprint.PrettyPrinter(indent=4)

# -- linalg --
import numpy as np
import torch as th
from einops import rearrange,repeat

# -- data mngmnt --
from pathlib import Path
from easydict import EasyDict as edict

# -- data --
import data_hub

# -- optical flow --
import svnlb

# -- caching results --
import cache_io

# -- network --
import dagl

# -- lightning module --
import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks import ModelCheckpoint

def run_exp(cfg):

    # -- create timer --
    timer = dagl.utils.timer.ExpTimer()

    # -- data --
    data,loaders = data_hub.sets.load(cfg)
    index = data.te.groups.index(cfg.vid_name)
    sample = data.te[index]

    # -- unpack --
    noisy,clean = sample['noisy'],sample['clean']

    # -- select color channel --
    if cfg.color == "bw":
        noisy = noisy[:,[0]].contiguous()
        clean = clean[:,[0]].contiguous()
    else:
        assert cfg.color == "rgb","must be rgb if not bw for now."

    # -- onto cuda --
    noisy,clean = noisy.to(cfg.device),clean.to(cfg.device)

    # -- normalize images --
    sigma = cfg.sigma/255.
    noisy /= 255.
    clean /= 255.

    # -- network --
    model = dagl.load_bw_deno_network(cfg.sigma).to(cfg.device)
    model.eval()

    # -- size --
    nframes = noisy.shape[0]
    ngroups = int(25 * 37./nframes)
    batch_size = ngroups*1024

    # -- optical flow --
    timer.start("flow")
    if cfg.comp_flow == "true":
        noisy_np = noisy.cpu().numpy()
        flows = svnlb.compute_flow(noisy_np,cfg.sigma)
        flows = edict({k:v.to(device) for k,v in flows.items()})
    else:
        flows = None
    timer.stop("flow")

    # -- internal adaptation --
    timer.start("adapt")
    run_internal_adapt = cfg.internal_adapt_nsteps > 0
    run_internal_adapt = run_internal_adapt and (cfg.internal_adapt_nepochs > 0)
    if run_internal_adapt:
        model.run_internal_adapt(noisy,cfg.sigma,flows=flows,
                                 ws=cfg.ws,wt=cfg.wt,batch_size=batch_size,
                                 nsteps=cfg.internal_adapt_nsteps,
                                 nepochs=cfg.internal_adapt_nepochs)
    timer.stop("adapt")

    # -- denoise --
    timer.start("deno")
    with th.no_grad():
        deno = model(noisy,0,ensemble=True)
        deno = deno.clamp(0.0, 1.0)
        deno = deno.detach()
        #flows=flows,ws=cfg.ws,wt=cfg.wt,batch_size=batch_size)
    timer.stop("deno")

    # -- save example --
    print("deno.shape: ",deno.shape)
    out_dir = Path(cfg.saved_dir) / str(cfg.uuid)
    deno_fns = dagl.utils.io.save_burst(deno,out_dir,"deno")

    # -- psnr --
    t = clean.shape[0]
    deno = deno.detach()
    clean = clean.reshape((t,-1))
    deno = deno.reshape((t,-1))
    mse = th.mean((clean - deno)**2,1)
    psnrs = -10. * th.log10(mse).detach()
    psnrs = list(psnrs.cpu().numpy())
    print(psnrs)

    # -- init results --
    results = edict()
    results.psnrs = psnrs
    results.deno_fn = deno_fns
    results.vid_name = [cfg.vid_name]
    for name,time in timer.items():
        results[name] = time

    return results


def default_cfg():
    # -- config --
    cfg = edict()
    cfg.nframes_tr = 5
    cfg.nframes_val = 5
    cfg.nframes_te = 0
    cfg.saved_dir = "./output/saved_results/"
    cfg.checkpoint_dir = "/home/gauenk/Documents/packages/dagl/output/checkpoints/"
    cfg.isize = None
    cfg.num_workers = 4
    cfg.device = "cuda:0"
    cfg.color = "bw"
    return cfg

def main():

    # -- (0) start info --
    verbose = True
    pid = os.getpid()
    print("PID: ",pid)

    # -- get cache --
    cache_dir = ".cache_io"
    cache_name = "test_rgb_net" # current!
    cache = cache_io.ExpCache(cache_dir,cache_name)
    cache.clear()

    # -- get mesh --
    # dnames = ["toy"]
    # vid_names = ["text_tourbus"]
    dnames = ["set8"]
    vid_names = ["snowboard","sunflower","tractor","hypersmooth",
                 "motorbike","park_joy","rafting","touchdown"]
    sigmas = [25]#[50,25,10]
    internal_adapt_nsteps = [0]
    internal_adapt_nepochs = [5]
    ws,wt = [29],[0]
    comp_flow = ["false"]
    exp_lists = {"dname":dnames,"vid_name":vid_names,"sigma":sigmas,
                 "internal_adapt_nsteps":internal_adapt_nsteps,
                 "internal_adapt_nepochs":internal_adapt_nepochs,
                 "comp_flow":comp_flow,"ws":ws,"wt":wt}
    exps = cache_io.mesh_pydicts(exp_lists) # create mesh
    # pp.pprint(exps)
    for exp in exps:
        if exp.internal_adapt_nsteps == 0:
            exp.internal_adapt_nepochs = 2

    # -- group with default --
    cfg = default_cfg()
    cache_io.append_configs(exps,cfg) # merge the two

    # -- run exps --
    nexps = len(exps)
    for exp_num,exp in enumerate(exps):

        # -- info --
        if verbose:
            print("-="*25+"-")
            print(f"Running experiment number {exp_num+1}/{nexps}")
            print("-="*25+"-")
            pp.pprint(exp)

        # -- logic --
        uuid = cache.get_uuid(exp) # assing ID to each Dict in Meshgrid
        results = cache.load_exp(exp) # possibly load result
        if results is None: # check if no result
            exp.uuid = uuid
            results = run_exp(exp)
            cache.save_exp(uuid,exp,results) # save to cache

    # -- load results --
    records = cache.load_flat_records(exps)

    # -- print by dname,sigma --
    for dname,ddf in records.groupby("dname"):
        field = "internal_adapt_nsteps"
        for adapt,adf in ddf.groupby(field):
            print("adapt: ",adapt)
            for sigma,sdf in adf.groupby("sigma"):
                ave_psnr,num_vids = 0,0
                for vname,vdf in sdf.groupby("vid_name"):
                    ave_psnr += vdf.psnrs.mean()
                    num_vids += 1
                ave_psnr /= num_vids
                print("[%d]: %2.3f" % (sigma,ave_psnr))


if __name__ == "__main__":
    main()
