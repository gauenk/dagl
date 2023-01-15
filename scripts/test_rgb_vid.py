
# -- misc --
import os,math,tqdm
import pprint,random,copy
pp = pprint.PrettyPrinter(indent=4)
from functools import partial

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
# import svnlb
from dagl import flow

# -- caching results --
import cache_io

# -- network --
import dagl
import dagl.configs as configs
from dagl import lightning
from dagl.utils.misc import optional,slice_flows
import dagl.utils.gpu_mem as gpu_mem
from dagl.utils.misc import rslice,write_pickle,read_pickle
from dagl.utils.proc_utils import get_fwd_fxn#spatial_chop,temporal_chop
from dagl.utils.aug_test import test_x8

def run_exp(_cfg):

    # -- init --
    cfg = copy.deepcopy(_cfg)
    cache_io.exp_strings2bools(cfg)

    # -- set device --
    th.cuda.empty_cache()
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
    nchnls = 1 if cfg.bw else 3
    model = dagl.load_model(cfg)
    model.eval()
    imax = 255.
    use_chop = (cfg.ca_fwd == "default") and (cfg.use_chop == "true")
    print("use_chop: ",use_chop)
    model.chop = use_chop

    # -- optional load trained weights --
    load_trained_state(model,cfg.use_train,cfg.ca_fwd,cfg.sigma,cfg.ws,cfg.wt)

    # -- data --
    data,loaders = data_hub.sets.load(cfg)
    groups = data.te.groups
    # indices = [i for i,g in enumerate(groups) if cfg.vid_name in g]
    indices = data_hub.filter_subseq(data.te,cfg.vid_name,
                                     cfg.frame_start,cfg.frame_end)

    for index in indices:

        # -- clean memory --
        th.cuda.empty_cache()
        print("index: ",index)

        # -- unpack --
        sample = data.te[index]
        region = sample['region']
        noisy,clean = sample['noisy'],sample['clean']
        noisy,clean = noisy.to(cfg.device),clean.to(cfg.device)
        vid_frames = sample['fnums'].numpy()
        print("[%d] noisy.shape: " % index,noisy.shape)
        temporal_chop = noisy.shape[0] > 20
        temporal_chop = temporal_chop and not(use_chop)

        # -- optional crop --
        noisy = rslice(noisy,region)
        clean = rslice(clean,region)
        print("[%d] noisy.shape: " % index,noisy.shape)

        # -- create timer --
        timer = dagl.utils.timer.ExpTimer()

        # -- optical flow --
        timer.start("flow")
        if cfg.flow is True:
            sigma_est = flow.est_sigma(noisy)
            flows = flow.run_batch(noisy[None,:],sigma_est)
        else:
            flows = flow.run_zeros(noisy[None,:])
        timer.sync_stop("flow")

        # -- fwd fxn --
        if cfg.aug_test:
            aug_fxn = partial(test_x8,model,use_refine=cfg.aug_refine_inds)
        else: aug_fxn = model
        fwd_fxn = get_fwd_fxn(cfg,aug_fxn)

        # -- run once for setup gpu --
        if cfg.burn_in is True:
            with th.no_grad():
                deno = fwd_fxn(noisy[[0],...,:128,:128]/imax,None)
            model.reset_times()

        # -- benchmarking --
        gpu_mem.print_peak_gpu_stats(False,"val",reset=True)
        timer.sync_start("deno")
        with th.no_grad():
            deno = fwd_fxn(noisy/imax,flows)
        deno = deno.clamp(0.,1.)*imax
        timer.sync_stop("deno")
        mem_alloc,mem_res = gpu_mem.print_peak_gpu_stats(True,"val",reset=True)

        # -- save example --
        out_dir = Path(cfg.saved_dir) / str(cfg.uuid)
        deno_fns = dagl.utils.io.save_burst(deno,out_dir,"deno")
        # dagl.utils.io.save_burst(clean,out_dir,"clean")

        # -- psnr --
        noisy_psnrs = dagl.utils.metrics.compute_psnrs(noisy,clean,div=imax)
        psnrs = dagl.utils.metrics.compute_psnrs(deno,clean,div=imax)
        ssims = dagl.utils.metrics.compute_ssims(deno,clean,div=imax)
        print(noisy_psnrs)
        print(psnrs)

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
            if not(name in results):
                results[name] = []
            results[name].append(time)
        for name,time in model.times.items():
            if not(name in results):
                results[name] = []
            results[name].append(time)
        print(model.times)

    th.cuda.empty_cache()
    return results

def load_trained_state(model,use_train,ca_fwd,sigma,ws,wt):

    # -- read cache --
    # results = cache.load_exp(cfg) # possibly load result
    if np.abs(sigma-50.) < 8:
        model_path = "output/checkpoints/1e61f94f-5d8a-4a9c-a873-401fff0de914-epoch=141.ckpt"
    elif np.abs(sigma-30.) < 11:
        model_path = "output/checkpoints/dd9c0d93-3cbf-4c1c-9cbb-f8cb677d1fed-epoch=33.ckpt"
    else:
        raise ValueError("Missing sigma level")

    # -- load model state --
    print("Loading trained network: ",model_path)
    state = th.load(model_path)['state_dict']
    lightning.remove_lightning_load_state(state)
    if hasattr(model,"model"):
        model.model.load_state_dict(state)
    else:
        model.load_state_dict(state)
    return model

def save_path_from_cfg(cfg):
    path = Path(cfg.dname) / cfg.vid_name
    train_str = "train" if  cfg.train == "true" else "notrain"
    path = path / "%s_%s" % (cfg.ca_fwd,train_str)

def main():

    # -- (0) start info --
    verbose = True
    pid = os.getpid()
    print("PID: ",pid)

    # -- get cache --
    cache_dir = ".cache_io"
    # cache_name = "test_rgb_net" # best results (aaai23)
    cache_name = "test_rgb_net_cvpr23" # best results
    cache = cache_io.ExpCache(cache_dir,cache_name)
    # cache.clear()

    # -- get defaults --
    cfg = configs.test_rgb_vid.default()
    cfg.isize = "512_512"
    # cfg.isize = "256_256"
    # cfg.isize = "128_128"
    # cfg.isize = "none"#"128_128"
    cfg.bw = True
    cfg.nframes = 5
    cfg.frame_start = 0
    cfg.frame_end = cfg.frame_start+cfg.nframes-1
    cfg.attn_mode = "dnls_k"
    cfg.burn_in = False

    # -- processing --
    cfg.spatial_crop_size = "none"
    cfg.spatial_crop_overlap = 0.#0.1
    cfg.temporal_crop_size = 7#cfg.nframes
    cfg.temporal_crop_overlap = 0/5.#4/5. # 3 of 5 frames

    # -- get mesh --
    dnames,sigmas = ["set8"],[30,50]#,30.]
    # vid_names = ["tractor"]
    # vid_names = ["sunflower"]
    # vid_names = ["sunflower","hypersmooth","tractor","motorbike"]
    # vid_names = ["park_joy"]
    vid_names = ["sunflower","tractor","park_joy"]
    # vid_names = ["sunflower","park_joy"]
    # vid_names = ["snowboard","sunflower","tractor","motorbike",
    #              "hypersmooth","park_joy","rafting","touchdown"]

    # -- standard --
    # ws,wt = [27],[3]
    # cfg.k_s = 100
    # cfg.k_a = 100
    # cfg.refine_inds = "f-f-f"

    # -- prop0 --
    cfg.k_s = 100
    cfg.k_a = 100
    # cfg.k_s = 75
    # cfg.k_a = 25
    cfg.ws = 27
    cfg.wt = 3
    cfg.return_inds = True
    cfg.arch_return_inds = True
    cfg.use_pfc = False
    # cfg.k_a = 50
    # cfg.refine_inds = "f-f-f"

    # -- prop1 --
    # ws,wt = ['27-27-3'],['3-3-0']
    # cfg.k_s = '100-500-100'
    # cfg.k_a = '100-100-100'
    # cfg.k_a = 50
    # cfg.refine_inds = "f-f-t"

    # -- prop2 --
    # ws,wt = ['27-27-27'],['3-3-3']
    # cfg.k_s = '100-100-100'
    # cfg.k_a = '100-100-100'
    # cfg.refine_inds = "f-f-f"

    flow = ["true"]
    ca_fwd_list,use_train = ["dnls_k"],["true"]
    # refine_inds = [("t-"*12)[:-1],("f-"*12)[:-1],("f-"+"t-"*11)[:-1]]
    refine_inds = [("t-"*12)[:-1],("f-"*12)[:-1],#("f-"+"t-"*11)[:-1],
                   ("f-f-f-t-"*3)[:-1],
                   ("f-t-t-t-"*3)[:-1]]
    # refine_inds = [("f-"*12)[:-1],("f-t-t-t-"*3)[:-1],
    #                ("f-"+"t-"*11)[:-1],("f-t-t-t-t-t-"*2)[:-1]]
    #,("f-f-t-t-"*3)[:-1],
    # ("f-f-f-t-"*3)[:-1]]#,("f-t-t-"*4)[:-1],("f-f-t-"*4)[:-1]]
    # print(refine_inds)
    # refine_inds = ["f-f-f","f-f-t","f-t-f","f-t-t"]
    # aug_test = ["false"]
    aug_test = ["true","false"]
    aug_refine_inds = ["true"]
    model_type = ['augmented']
    ws_r = [1]
    # ws_r = [1,3]
    exp_lists = {"dname":dnames,"vid_name":vid_names,"sigma":sigmas,
                 "flow":flow,"use_train":use_train,
                 "ca_fwd":ca_fwd_list,"use_chop":["false"],"model_type":model_type,
                 "refine_inds":refine_inds,"ws_r":ws_r,
                 "aug_refine_inds":aug_refine_inds,"aug_test":aug_test}
    exps_a = cache_io.mesh_pydicts(exp_lists) # create mesh
    cache_io.append_configs(exps_a,cfg) # merge the two

    # cfg.attn_mode = "dnls_k"
    # cfg.ws = "27-5-5"
    # cfg.wt = "3-0-0"
    # cfg.k_s = '500-500-100'
    # cfg.k_a = 100
    # cfg.refine_inds = "f-t-t"
    # exps_a = cache_io.mesh_pydicts(exp_lists) # create mesh
    # cache_io.append_configs(exps_a,cfg) # merge the two

    # -- original w/out training --
    cfg.ws = 27
    exp_lists['refine_inds'] = ['na']
    exp_lists['model_type'] = ['original']
    exp_lists['flow'] = ["false"]
    exp_lists['use_train'] = ["false"]#,"true"]
    exp_lists['ca_fwd'] = ["default"]
    exp_lists['use_chop'] = ["false"]
    exps_b = cache_io.mesh_pydicts(exp_lists) # create mesh
    cfg.bw = True
    cfg.spatial_crop_size = 256
    cache_io.append_configs(exps_b,cfg) # merge the two

    # -- cat exps --
    exps = exps_a# + exps_b

    # -- run exps --
    nexps = len(exps)
    for exp_num,exp in enumerate(exps):

        # -- info --
        if verbose:
            print("-="*25+"-")
            print(f"Running experiment number {exp_num+1}/{nexps}")
            print("-="*25+"-")
            pp.pprint(exp)
        # exit(0)

        # -- logic --
        uuid = cache.get_uuid(exp) # assing ID to each Dict in Meshgrid

        clear_exp = exp.attn_mode == "dnls_k" and exp.model_type == "refactored"
        clear_exp = clear_exp and (exp.ws != 27)
        clear_exp = clear_exp or ('t' in exp.refine_inds)
        # cache.clear_exp(uuid)
        # if clear_exp or True:
        #     cache.clear_exp(uuid)
        # if exp.use_chop == "false" and exp.ca_fwd != "dnls_k":
        #     cache.clear_exp(uuid)
        # if exp.ca_fwd != "dnls_k" and exp.sigma == 30.:
        #     cache.clear_exp(uuid)
        # if exp.sigma == 30. and exp.ca_fwd == "dnls_k":
        #     cache.clear_exp(uuid)
        # if exp.sigma == 10. and exp.ca_fwd == "dnls_k":
        #     cache.clear_exp(uuid)
        results = cache.load_exp(exp) # possibly load result
        if results is None: # check if no result
            exp.uuid = uuid
            results = run_exp(exp)
            cache.save_exp(uuid,exp,results) # save to cache

    # -- load results --
    records = cache.load_flat_records(exps)
    # print(records.filter(like="timer"))

    # -- neat report --
    fields = ['use_train','refine_inds','use_chop','aug_test','sigma','vid_name']
    fields_summ = ['use_train','refine_inds','use_chop','aug_test']
    res_fields = ['psnrs','ssims','timer_deno','timer_extract',
                  'timer_search','timer_agg','mem_alloc','mem_res']
    res_fmt = ['%2.3f','%1.3f','%2.3f','%2.3f','%2.3f','%2.3f','%2.3f','%2.3f']

    # -- run agg --
    agg = {key:[np.stack] for key in res_fields}
    grouped = records.groupby(fields).agg(agg)
    grouped.columns = res_fields
    grouped = grouped.reset_index()
    for field in res_fields:
        grouped[field] = grouped[field].apply(np.mean)
    res_fmt = {k:v for k,v in zip(res_fields,res_fmt)}
    print("\n\n" + "-="*10+"-" + "Report" + "-="*10+"-")
    for sigma,sdf in grouped.groupby("sigma"):
        print("--> sigma: %d <--" % sigma)
        for gfields,gdf in sdf.groupby(fields_summ):
            gfields = list(gfields)

            # -- header --
            header = "-"*15 + "\n"
            for i,field in enumerate(fields_summ):
                header += "%s " % (gfields[i])
            print(header)

            for vid_name,vdf in gdf.groupby("vid_name"):
                # -- res --
                res = "%13s: " % vid_name
                for field in res_fields:
                    res += res_fmt[field] % (vdf[field].mean()) + " "
                print(res)
            # -- res --
            res = "%13s: " % "Ave"
            for field in res_fields:
                res += res_fmt[field] % (gdf[field].mean()) + " "
            print(res)


if __name__ == "__main__":
    main()
