
# -- misc --
import os,math,tqdm
import pprint,random
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
from dagl import flow

# -- caching results --
import cache_io

# -- network --
import dagl
import dagl.configs.test_rgb_vid as configs
from dagl import lightning
from dagl.utils.misc import optional
import dagl.utils.gpu_mem as gpu_mem
from dagl.utils.misc import rslice,write_pickle,read_pickle
from dagl.utils.proc_utils import get_fwd_fxn

def run_exp(cfg):

    # -- set device --
    th.cuda.set_device(int(cfg.device.split(":")[1]))

    # -- set seed --
    configs.set_seed(cfg.seed)

    # -- init results --
    results = edict()
    results.psnrs = []
    results.ssims = []
    results.noisy_psnrs = []
    results.deno_fns = []
    results.vid_frames = []
    results.vid_name = []
    results.timer_flow = []
    results.timer_deno = []
    results.mem_res = []
    results.mem_alloc = []

    # -- network --
    model_cfg = dagl.extract_model_config(cfg)
    model = dagl.load_model(model_cfg)
    model.eval()
    imax = 255.

    # -- optional load trained weights --
    # load_trained_state(model,cfg.sigma,cfg.use_train)

    # -- data --
    data,loaders = data_hub.sets.load(cfg)
    frame_start = optional(cfg,"frame_start",0)
    frame_end = optional(cfg,"frame_end",-1)
    indices = data_hub.filter_subseq(data[cfg.dset],cfg.vid_name,frame_start,frame_end)

    # # -- optional filter --
    # groups = data.te.groups
    # indices = [i for i,g in enumerate(groups) if cfg.vid_name in g]
    # if frame_start >= 0 and frame_end > 0:
    #     def fbnds(fnums,lb,ub): return (lb <= np.min(fnums)) and (ub >= np.max(fnums))
    #     indices = [i for i in indices if fbnds(data.te.paths['fnums'][groups[i]],
    #                                            cfg.frame_start,cfg.frame_end)]

    for index in indices:

        # -- clean memory --
        th.cuda.empty_cache()
        print("index: ",index)

        # -- unpack --
        sample = data[cfg.dset][index]
        region = sample['region']
        noisy,clean = sample['noisy'],sample['clean']
        noisy,clean = noisy.to(cfg.device),clean.to(cfg.device)
        vid_frames = sample['fnums'].cpu().numpy()
        print("[%d] noisy.shape: " % index,noisy.shape)

        # -- optional crop --
        noisy = rslice(noisy,region)
        clean = rslice(clean,region)
        print("[%d] noisy.shape: " % index,noisy.shape)

        # -- create timer --
        timer = dagl.utils.timer.ExpTimer()

        # -- size --
        nframes = noisy.shape[0]
        ngroups = int(25 * 37./nframes)
        batch_size = 390*39#ngroups*1024

        # -- optical flow --
        timer.start("flow")
        if cfg.flow == "true":
            sigma_est = flow.est_sigma(noisy)
            flows = flow.run_batch(noisy[None,:],sigma_est)
        else:
            flows = flow.run_zeros(noisy[None,:])
        timer.stop("flow")

        # -- denoise --
        fwd_fxn = get_fwd_fxn(cfg,model)
        # tsize = 10
        timer.start("deno")
        gpu_mem.print_peak_gpu_stats(False,"val",reset=True)
        with th.no_grad():
            # deno = model(noisy/imax,flows)
            deno = fwd_fxn(noisy/imax,flows)
        deno = th.clamp(deno,0,1.)*imax
        timer.stop("deno")
        mem_alloc,mem_res = gpu_mem.print_peak_gpu_stats(True,"val",reset=True)
        deno = deno.clamp(0.,imax)
        print("deno.shape: ",deno.shape)

        # -- save example --
        out_dir = Path(cfg.saved_dir) / str(cfg.uuid)
        deno_fns = dagl.utils.io.save_burst(deno,out_dir,"deno")

        # -- psnr --
        noisy_psnrs = dagl.utils.metrics.compute_psnrs(noisy,clean,div=imax)
        psnrs = dagl.utils.metrics.compute_psnrs(deno,clean,div=imax)
        ssims = dagl.utils.metrics.compute_ssims(deno,clean,div=imax)
        print(noisy_psnrs)
        print(psnrs,psnrs.mean())

        # -- append results --
        results.psnrs.append(psnrs)
        results.ssims.append(ssims)
        results.noisy_psnrs.append(noisy_psnrs)
        results.deno_fns.append(deno_fns)
        results.vid_frames.append(vid_frames)
        results.vid_name.append([cfg.vid_name])
        results.mem_res.append([mem_res])
        results.mem_alloc.append([mem_alloc])
        for name,time in timer.items():
            results[name].append(time)

    return results

def main():

    # -- (0) start info --
    verbose = True
    pid = os.getpid()
    print("PID: ",pid)

    # -- get cache --
    cache_dir = ".cache_io"
    cache_name = "baseline" # current!
    cache = cache_io.ExpCache(cache_dir,cache_name)
    # cache.clear()

    # -- get defaults --
    cfg = configs.default()
    cfg.seed = 123

    # -- data chop --
    cfg.nframes = 6
    cfg.isize = "256_256"
    cfg.cropmode = "center"
    cfg.frame_start = 0
    cfg.frame_end = cfg.frame_start+cfg.nframes-1
    cfg.frame_end = 0 if cfg.frame_end < 0 else cfg.frame_end

    # -- processing --
    cfg.spatial_crop_size = "none"
    cfg.spatial_crop_overlap = 0.#0.1
    cfg.temporal_crop_size = cfg.nframes
    cfg.temporal_crop_overlap = 0/5.#4/5. # 3 of 5 frames
    cfg.ps = 10

    # -- get mesh --
    k,bs,stride = [7],[28*1024],[5]
    # ws,wt,k,bs,stride = [20],[0],[7],[28*1024],[5]
    # ws,wt,k,bs,stride = [29],[3],[7],[28*1024],[5]
    # sigmas = [10.,30.]
    # sigmas = [30.,50.]
    sigmas = [25.]
    # sigmas = [30.]
    # sigmas = [50.]
    # ws,wt = [29],[3]
    # sigmas = [50.]
    ws,wt = [29],[3]
    dnames,use_train = ["set8"],["true","false"]
    # sigmas = [50.]
    # ws,wt,k,bs,stride = [15],[3],[7],[32],[5]
    # wt,sigmas = [0],[30.]
    # vid_names = ["tractor"]
    # bs = [512*512]
    # vid_names = ["sunflower"]#,"hypersmooth","tractor"]
    vid_names = ["rafting"]
    # vid_names = ["rafting","sunflow"]
    # vid_names = ["sunflower","tractor","snowboard","motorbike",
    #              "hypersmooth","park_joy","rafting","touchdown"]
    flow = ["true"]
    # flow = ["true","false"]
    model_names = ["refactored"]
    exp_lists = {"dname":dnames,"vid_name":vid_names,"sigma":sigmas,
                 "flow":flow,"ws":ws,"wt":wt,"use_train":use_train,"stride":stride,
                 "ws":ws,"wt":wt,"k":k, "bs":bs, "model_name":model_names}
    exps_a = cache_io.mesh_pydicts(exp_lists) # create mesh
    # exp_lists['wt'] = [3]
    # exp_lists['bs'] = [512*512//8]
    # exps_a = cache_io.mesh_pydicts(exp_lists)
    # exp_lists['ws'] = [29]
    # exp_lists['wt'] = [0]
    # exps_a += cache_io.mesh_pydicts(exp_lists) # create mesh
    cache_io.append_configs(exps_a,cfg) # merge the two


    # -- original w/out training --
    exp_lists['model_name'] = ["original"]
    exp_lists['flow'] = ["false"]
    exp_lists['use_train'] = ["false"]#,"true"]
    exps_b = cache_io.mesh_pydicts(exp_lists) # create mesh
    cache_io.append_configs(exps_b,cfg) # merge the two

    # -- cat exps --
    exps = exps_a + exps_b
    # exps = exps_b

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
        cache.clear_exp(uuid)
        # if exp.model_name != "refactored":
        #     cache.clear_exp(uuid)
        results = cache.load_exp(exp) # possibly load result
        if results is None: # check if no result
            exp.uuid = uuid
            results = run_exp(exp)
            cache.save_exp(uuid,exp,results) # save to cache

    # -- load results --
    records = cache.load_flat_records(exps)
    # print(records[['timer_deno','model_name','mem_res']])
    # exit(0)
    # print(records)
    # print(records.filter(like="timer"))

    # -- viz report --
    for use_train,tdf in records.groupby("use_train"):
        for wt,wdf in tdf.groupby("wt"):
            for sigma,sdf in wdf.groupby("sigma"):
                print("----- %d -----" % sigma)
                for ca_group,gdf in sdf.groupby("model_name"):
                    for use_flow,fdf in gdf.groupby("flow"):
                        agg_psnrs,agg_ssims,agg_dtime = [],[],[]
                        agg_mem_res,agg_mem_alloc = [],[]
                        print("--- %s (%s,%s,%s) ---" % (ca_group,use_train,use_flow,wt))
                        for vname,vdf in fdf.groupby("vid_name"):
                            psnrs = np.stack(vdf['psnrs'])
                            dtime = np.stack(vdf['timer_deno'])
                            mem_res = np.stack(vdf['mem_res'])
                            mem_alloc = np.stack(vdf['mem_alloc'])
                            ssims = np.stack(vdf['ssims'])
                            psnr_mean = psnrs.mean().item()
                            ssim_mean = ssims.mean().item()
                            uuid = vdf['uuid'].iloc[0]

                            # print(dtime,mem_gb)
                            # print(vname,psnr_mean,ssim_mean,uuid)
                            print("%13s: %2.3f %1.3f %s" % (vname,psnr_mean,ssim_mean,uuid))
                            agg_psnrs.append(psnr_mean)
                            agg_ssims.append(ssim_mean)
                            agg_mem_res.append(mem_res.mean().item())
                            agg_mem_alloc.append(mem_alloc.mean().item())
                            agg_dtime.append(dtime.mean().item())
                        psnr_mean = np.mean(agg_psnrs)
                        ssim_mean = np.mean(agg_ssims)
                        dtime_mean = np.mean(agg_dtime)
                        mem_res_mean = np.mean(agg_mem_res)
                        mem_alloc_mean = np.mean(agg_mem_alloc)
                        uuid = tdf['uuid']
                        params = ("Ave",psnr_mean,ssim_mean,dtime_mean,
                                  mem_res_mean,mem_alloc_mean)
                        print("%13s: %2.3f %1.3f %2.3f %2.3f %2.3f" % params)


if __name__ == "__main__":
    main()