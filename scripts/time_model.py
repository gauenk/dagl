

# -- misc --
import os,math,tqdm
import pprint,random,copy
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

# -- profiling --
import torch.autograd.profiler as profiler
from torch.profiler import profile, record_function, ProfilerActivity


# -- network --
import dagl
import dagl.configs.test_rgb_vid as configs
from dagl import lightning
from dagl.utils.misc import optional
import dagl.utils.gpu_mem as gpu_mem
from dagl.utils.misc import rslice,write_pickle,read_pickle
from dagl.utils.proc_utils import get_fwd_fxn

def run_exp(_cfg):

    # -- set device --
    cfg = copy.deepcopy(_cfg)
    cache_io.exp_strings2bools(cfg)
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
    load_trained_state(model,cfg.sigma,cfg.use_train)

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

        # -- init & warm-up --
        # layer_timer = dagl.hook_timer_to_model(model)
        fwd_fxn = get_fwd_fxn(cfg,model)
        # rands = th.randn_like(noisy).clamp(0,1)
        # _ = fwd_fxn(rands,flows) # warp-up

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

        # -- profiler --
        with th.no_grad():
            with profiler.profile(ProfilerActivity.CUDA, with_stack=True, profile_memory=True, with_modules=True, use_cuda=True) as prof:
                with record_function("model_inference"):
                    out = model(noisy/imax,flows)
                # out = fwd_fxn(noisy/imax,flows)
        # kave = prof.key_averages(group_by_stack_n=10).table(sort_by='self_cpu_time_total')
        kave = prof.key_averages(group_by_stack_n=10).table(sort_by='cuda_time_total',row_limit=8)
        print(kave)
        # print(prof.key_averages(group_by_stack_n=5).table(
        # print(prof.key_averages(group_by_stack_n=5).table(
        #     sort_by='cuda_time_total', row_limit=20))
        # prof.export_chrome_trace("trace.json")


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

def load_trained_state(model,sigma,use_train):
    if not(use_train): return
    model_path = ""
    if np.abs(sigma - 25.) < 20:#1e-10:
        model_path = Path("output/checkpoints/")
        model_path /= "dd9c0d93-3cbf-4c1c-9cbb-f8cb677d1fed-epoch=32.ckpt"
    print("using trained path: ",model_path)
    state = th.load(model_path)['state_dict']
    lightning.remove_lightning_load_state(state)
    model.load_state_dict(state)
    return model

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
    cfg.nframes = 5
    cfg.isize = "256_256"
    # cfg.isize = "128_128"
    cfg.cropmode = "center"
    cfg.frame_start = 5
    cfg.frame_end = cfg.frame_start+cfg.nframes-1
    cfg.frame_end = 0 if cfg.frame_end < 0 else cfg.frame_end

    # -- processing --
    cfg.spatial_crop_size = "none"
    cfg.spatial_crop_overlap = 0.#0.1
    cfg.temporal_crop_size = cfg.nframes
    cfg.temporal_crop_overlap = 0/5.#4/5. # 3 of 5 frames
    cfg.ps = 7
    cfg.k_s = 64
    cfg.k_a = 64
    cfg.bs = -1#28*1024

    # -- get mesh --
    # ws,wt,k,bs,stride = [20],[0],[7],[28*1024],[5]
    # ws,wt,k,bs,stride = [29],[3],[7],[28*1024],[5]
    # sigmas = [10.,30.]
    # sigmas = [30.,50.]
    sigmas = [25.]
    # sigmas = [30.]
    # sigmas = [50.]
    # sigmas = [50.]
    # ws,wt = [29],[3]
    ws,wt = [21],[3]
    # ws,wt = [15],[3]
    # ws,wt = [8],[3]
    dnames = ["set8"]
    # sigmas = [50.]
    # ws,wt,k,bs,stride = [15],[3],[7],[32],[5]
    # wt,sigmas = [0],[30.]
    # vid_names = ["motorbike"]
    # vid_names = ["park_joy"]
    # vid_names = ["tractor"]
    # vid_names = ["snowboard"]
    vid_names = ["sunflower"]#,"hypersmooth","tractor"]
    # vid_names = ["rafting"]
    # vid_names = ["rafting","sunflower"]
    # vid_names = ["sunflower","tractor","snowboard","motorbike",
    #              "hypersmooth","park_joy","rafting","touchdown"]
    use_train = ["true"]
    flow = ["true"]
    # flow = ["true","false"]
    model_type = ["augmented"]
    exp_lists = {"dname":dnames,"vid_name":vid_names,"sigma":sigmas,
                 "flow":flow,"ws":ws,"wt":wt,"use_train":use_train,
                 "ws":ws,"wt":wt,"model_type":model_type}
    exps_a = cache_io.mesh_pydicts(exp_lists) # create mesh
    # exp_lists['wt'] = [3]
    # exp_lists['bs'] = [512*512//8]
    # exps_a = cache_io.mesh_pydicts(exp_lists)
    exp_lists['ws'] = [8]
    exp_lists['wt'] = [0]
    exps_a += cache_io.mesh_pydicts(exp_lists) # create mesh
    cache_io.append_configs(exps_a,cfg) # merge the two



    # -- original w/out training --
    exp_lists['ws'] = [21]
    exp_lists['model_type'] = ["original"]
    exp_lists['flow'] = ["false"]
    exp_lists['use_train'] = ["false"]#,"true"]
    exps_b = cache_io.mesh_pydicts(exp_lists) # create mesh
    cache_io.append_configs(exps_b,cfg) # merge the two

    # -- cat exps --
    # exps = exps_a + exps_b
    exps = exps_a
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
        if exp.use_train == "true":
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
    print(records)

if __name__ == "__main__":
    main()
