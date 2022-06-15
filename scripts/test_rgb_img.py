
# -- misc --
import os,math,tqdm
import pprint
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
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
from torch.autograd import Variable

# -- lightning module --
import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks import ModelCheckpoint

def run_exp(cfg):

    # -- init results --
    results = edict()
    results.psnrs = []
    results.deno_fn = []
    results.names = []

    # -- data --
    data,loaders = data_hub.sets.load(cfg)
    loader = iter(loaders.te)

    # -- network --
    model = dagl.load_bw_deno_network(cfg.sigma).to(cfg.device)
    model.eval()

    # -- for each batch --
    for batch in loader:

        # -- unpack --
        noisy,clean = batch['noisy'],batch['clean']
        name = data.te.groups[int(batch['index'][0])]
        print("noisy.shape: ",noisy.shape)

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

        # -- size --
        nframes = noisy.shape[0]
        ngroups = int(25 * 37./nframes)
        batch_size = ngroups*1024

        # -- optical flow --
        noisy_np = noisy.cpu().numpy()
        if cfg.comp_flow == "true":
            flows = svnlb.compute_flow(noisy_np,cfg.sigma)
            flows = edict({k:v.to(device) for k,v in flows.items()})
        else:
            flows = None

        # -- internal adaptation --
        run_internal_adapt = cfg.internal_adapt_nsteps > 0
        run_internal_adapt = run_internal_adapt and (cfg.internal_adapt_nepochs > 0)
        if run_internal_adapt:
            model.run_internal_adapt(noisy,cfg.sigma,flows=flows,
                                     ws=cfg.ws,wt=cfg.wt,batch_size=batch_size,
                                     nsteps=cfg.internal_adapt_nsteps,
                                     nepochs=cfg.internal_adapt_nepochs)
        # -- denoise --
        with th.no_grad():
            deno = model(noisy,0,ensemble=True)
            deno = deno.clamp(0.0, 1.0)
            deno = deno.detach()

        # -- save example --
        out_dir = Path(cfg.saved_dir)# / str(cfg.uuid)
        if not out_dir.exists(): out_dir.mkdir(parents=True)
        deno_fn = out_dir / ("deno_%s.png" % name)
        # clean_fn = out_dir / ("clean_%s.png" % name)
        # noisy_fn = out_dir / ("noisy_%s.png" % name)
        dagl.utils.io.save_image(deno[0],deno_fn)
        # dagl.utils.io.save_image(clean[0],clean_fn)
        # dagl.utils.io.save_image(noisy[0],noisy_fn)

        # -- psnr --
        noisy_psnr = -10. * th.log10(th.mean((clean - noisy)**2)).item()
        print(noisy_psnr)
        psnr = -10. * th.log10(th.mean((clean - deno)**2)).item()
        print(psnr)

        # -- append results --
        results.psnrs.append(psnr)
        results.deno_fn.append([deno_fn])
        results.names.append([name])

    print(results)
    return results

def rescale(img):
    if th.is_tensor(img):
        return th.clamp((img * 255.),0.,255.).type(th.uint8)
    else:
        return np.clip((img * 255.),0.,255.).astype(np.uint8)

def default_cfg():
    # -- config --
    cfg = edict()
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
    dnames = ["bsd68"]
    sigmas = [25,10]
    internal_adapt_nsteps = [0]
    internal_adapt_nepochs = [5]
    ws,wt = [29],[0]
    comp_flow = ["false"]
    exp_lists = {"dname":dnames,"sigma":sigmas,
                 "internal_adapt_nsteps":internal_adapt_nsteps,
                 "internal_adapt_nepochs":internal_adapt_nepochs,
                 "comp_flow":comp_flow,"ws":ws,"wt":wt}
    exps = cache_io.mesh_pydicts(exp_lists) # create mesh

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
                ave_psnr = sdf.psnrs.mean()
                print("[%d]: %2.3f" % (sigma,ave_psnr))


if __name__ == "__main__":
    main()
