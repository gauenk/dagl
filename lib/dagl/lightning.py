

# -- misc --
import os,math,tqdm
import pprint,copy
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
from dagl import flow

# -- caching results --
import cache_io

# -- network --
import dagl
import dagl.utils.io as io
import dagl.configs as configs
import dagl.utils.gpu_mem as gpu_mem
from dagl.utils.timer import ExpTimer
from dagl.utils.metrics import compute_psnrs,compute_ssims
from dagl.utils.misc import rslice,write_pickle,read_pickle

# -- generic logging --
import logging
logging.basicConfig()

# -- lightning module --
import torch
import pytorch_lightning as pl
from pytorch_lightning import Callback
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.utilities.distributed import rank_zero_only

# -- local --
from .utils import optional

def grab_grad(model):
    for param in model.parameters():
        if hasattr(param,'weight'):
            print(param.weight.grad)
            exit(0)

class DaglLit(pl.LightningModule):

    def __init__(self,model_cfg,flow=True,isize=None,batch_size=32,
                 lr_init=0.0002,weight_decay=0.02,nepochs=250,
                 scheduler="default",task="denoise",uuid="default"):
        super().__init__()

        # -- meta params --
        self.flow = flow
        self.isize = isize
        self.uuid = uuid

        # -- learning --
        self.batch_size = batch_size
        self.lr_init = lr_init
        self.weight_decay = weight_decay
        self.nepochs = nepochs
        self.scheduler = scheduler
        self.task = task

        # -- load model --
        self.net = dagl.load_model(model_cfg)
        self.net.train()
        # self.net._apply_freeze()

        # -- set logger --
        self.gen_loger = logging.getLogger('lightning')
        self.gen_loger.setLevel("NOTSET")
        # self.attn_mode = model_cfg['attn_mode']

        # -- manual optim --
        # self.automatic_optimization = False
        # self.set_backward_hooks()
        # print(list(self.net.output_proj.proj.modules())[0])
        # exit(0)

    def forward(self,vid,clamp=False):
        B = vid.shape[0]
        flows = self._get_flow(vid)
        deno = []
        for b in range(B):
            flows_b = flow.bindex(flows,b,True)
            deno_b = self.net(vid[b],flows=flows_b)
            deno.append(deno_b)
        deno = th.stack(deno)
        if clamp:
            deno = th.clamp(deno,0.,1.)
        return deno

    def _get_flow(self,vid):
        if self.flow:
            flows = flow.run_batch(vid)
        else:
            flows = flow.run_zeros(vid)
        return flows

    def get_default_optim(self):
        optim = th.optim.AdamW(self.parameters(),
                               lr=self.lr_init, betas=(0.9, 0.999),
                               eps=1e-8, weight_decay=self.weight_decay)
        scheduler_cosine = th.optim.lr_scheduler.CosineAnnealingLR(optim,
                                self.nepochs, eta_min=1e-6)
        scheduler = scheduler_cosine
        return optim, scheduler

    def get_steplr_optim(self):
        optim = th.optim.AdamW(self.parameters(),
                               lr=self.lr_init, betas=(0.9, 0.999),
                               eps=1e-8, weight_decay=self.weight_decay)
        step_size = 5
        scheduler = th.optim.lr_scheduler.StepLR(optim, step_size, gamma=0.5,
                                                 last_epoch=-1)
        return optim, scheduler

    def configure_optimizers(self):
        if self.scheduler == "default":
            optim,scheduler = self.get_default_optim()
        elif self.scheduler == "step_lr":
            optim,scheduler = self.get_steplr_optim()
        else:
            raise ValueError("Uknown scheduler.")
        return [optim], [scheduler]

    def training_step(self, batch, batch_idx):

        # -- init --
        # opt = self.optimizers()
        # opt.zero_grad()

        # -- each sample in batch --
        noisy_key = self.get_data_keys()[0]
        loss = 0 # init @ zero
        nbatch = len(batch[noisy_key])
        max_bs = -1#self.net.max_batch_size
        bsize = nbatch if max_bs == -1 else max_bs
        ngroups = (nbatch-1)//bsize+1
        denos,cleans = [],[]
        for i in range(ngroups):
            # th.cuda.empty_cache()
            binds = slice(i*bsize,(i+1)*bsize)
            deno_i,clean_i,loss_i = self.training_step_i(batch, binds)
            loss += loss_i
            denos.append(deno_i)
            cleans.append(clean_i)
        loss = loss / nbatch
        denos = th.cat(denos)
        cleans = th.cat(cleans)

        # -- append --
        # grab_grad(self.net)

        # -- log --
        self.log("train_loss", loss.item(), on_step=True,
                 on_epoch=False,batch_size=self.batch_size)

        # -- terminal log --
        psnr = np.mean(compute_psnrs(denos,cleans,div=1.)).item()
        self.gen_loger.info("train_psnr: %2.2f" % psnr)
        # print("train_psnr: %2.2f" % val_psnr)
        self.log("train_psnr", psnr, on_step=True,
                 on_epoch=False, batch_size=self.batch_size)

        # -- proj --
        # for name0,mod0 in self.net.output_proj.proj.named_children():
        #     print(name0)
        #     for name,param in mod0.named_parameters():
        #         if param is None: continue
        #         data = param.data
        #         pmin = data.min().item()
        #         pmax = data.max().item()
        #         print(name,data.shape,th.isnan(th.any(data)),pmin,pmax)

        # -- scheduler step --
        # sch = self.lr_schedulers()
        # sch.step()

        # -- optimizer step --
        # print("pre" + "\n"*10)
        # self.inspect_param_grads()
        # loss.backward()
        # self.manual_backward(loss)
        # print("post" + "\n"*10)
        # self.inspect_param_grads()
        # opt.step()

        return loss


    def set_backward_hooks(self):
        def fwd_hook(name, module, input, output):
            any_nan = False
            # for g in gradInput:
            #     if g is None: continue
            if isinstance(input,tuple):
                for _input in input:
                    if _input is None: continue
                    if isinstance(_input,int): continue
                    any_nan = any_nan or th.any(th.isnan(_input))
            else:
                any_nan = any_nan or th.any(th.isnan(input))
            if isinstance(output,tuple):
                for _output in output:
                    if _output is None: continue
                    if isinstance(_output,int): continue
                    any_nan = any_nan or th.any(th.isnan(_output))
            else:
                any_nan = any_nan or th.any(th.isnan(output))

            if any_nan:
                print("fwd.")
                print(name)
                print(module)
                if isinstance(input,tuple):
                    for _input in input:
                        if _input is None: continue
                        if isinstance(_input,int): continue
                        print(_input.shape,th.any(th.isnan(_input)))
                else:
                    print(input.shape,th.any(th.isnan(input)))
                if isinstance(output,tuple):
                    for _output in output:
                        if _output is None: continue
                        if isinstance(_output,int): continue
                        print(_output.shape,th.any(th.isnan(_output)))
                else:
                    print(output.shape,th.any(th.isnan(output)))
                    # any_nan = any_nan or th.any(th.isnan(output))
                # print(input)
                # print(output)
                # print(input.shape)
                # print(output.shape)
                # print("shapes: ")
                # for g in gradInput:
                #     print(g.shape,th.any(th.isnan(g)))
                # for g in gradOutput:
                #     print(g.shape)
                exit(0)

        def bwd_hook(name, module, gradInput, gradOutput):
            any_nan = False
            for g in gradInput:
                if g is None: continue
                any_nan = any_nan or th.any(th.isnan(g))
            if any_nan:
                print("bwd.")
                print(name)
                print(module)
                print(gradInput)
                print(gradOutput)
                print("shapes: ")
                for g in gradInput:
                    print(g.shape,th.any(th.isnan(g)))
                for g in gradOutput:
                    print(g.shape,th.any(th.isnan(g)))
                exit(0)
        for name,mod in self.named_modules():
            if name == "net": continue
            named_hook = partial(fwd_hook,name)
            mod.register_forward_hook(named_hook)
            named_hook = partial(bwd_hook,name)
            mod.register_backward_hook(named_hook)
        #     print(name)
        # exit(0)
        # for param in self.parameters():
        #     param.backward(register_backward_hook(hook))

    def inspect_param_grads(self):
        for param in self.parameters():
            if param.grad is None: continue
            any_nan = th.any(th.isnan(param.grad))
            pmin = param.grad.min().item()
            pmax = param.grad.max().item()
            print("[any,pmin,pmax]: ",any_nan,pmin,pmax)

    def training_step_i(self, batch, batch_inds):

        # -- unpack batch
        noisy_key,clean_key = self.get_data_keys()
        noisy = batch[noisy_key][batch_inds]/255.
        clean = batch[clean_key][batch_inds]/255.
        # region = batch['region'][[i]]

        # -- get data --
        # B = noisy.shape[0]
        # for b in range(B):
        #     noisy[b] = rslice(noisy,region)
        #     clean[b] = rslice(clean,region)
        # print("noisy.shape: ",noisy.shape)

        # -- foward --
        deno = self.forward(noisy,False)
        # print("deno.shape: ",deno.shape)

        # -- save a few --
        if self.global_step % 300 == 0: # and i == 0
            out_dir = "./output/lightning/%s/%06d" % (self.uuid,self.global_step)
            B = deno.shape[0]
            for b in range(B):
                io.save_burst(deno[b],out_dir,"deno_%d" % b)
                io.save_burst(noisy[b],out_dir,"noisy_%d" % b)
                io.save_burst(clean[b],out_dir,"clean_%d" % b)

        # -- get loss for train types --
        if self.task == "deblur":
            eps = 1.*1e-3
            diff = th.sqrt((clean - deno)**2 + eps**2)
            loss = th.mean(diff)
        else:
            loss = th.mean((clean - deno)**2)

        return deno.detach(),clean,loss


    def get_data_keys(self):
        if self.task == "deblur":
            return "blur","sharp"
        else:
            return "noisy","clean"

    def validation_step(self, batch, batch_idx):

        # -- unpack data --
        noisy_key,clean_key = self.get_data_keys()
        noisy,clean = batch[noisy_key][[0]]/255.,batch[clean_key][[0]]/255.
        # region = batch['region'][0]
        # noisy[0] = rslice(noisy[0],region)
        # clean[0] = rslice(clean[0],region)

        # -- forward --
        gpu_mem.print_peak_gpu_stats(False,"val",reset=True)
        with th.no_grad():
            deno = self.forward(noisy,True)
        mem_res,mem_alloc = gpu_mem.print_peak_gpu_stats(False,"val",reset=True)

        # -- loss --
        loss = th.mean((clean - deno)**2)
        _loss = loss.item()

        # -- report --
        self.log("val_loss", _loss, on_step=False,
                 on_epoch=True,batch_size=1, sync_dist=True)
        self.log("val_mem_res", mem_res, on_step=False,
                 on_epoch=True,batch_size=1, sync_dist=True)
        self.log("val_mem_alloc", mem_alloc, on_step=False,
                 on_epoch=True,batch_size=1, sync_dist=True)

        # -- terminal log --
        val_psnr = np.mean(compute_psnrs(deno,clean,div=1.)).item()
        self.gen_loger.info("val_psnr: %2.2f" % val_psnr)

    def test_step(self, batch, batch_nb):

        # -- denoise --
        index,region = batch['index'][0],batch['region'][0]
        noisy_key,clean_key = self.get_data_keys()
        noisy,clean = batch[noisy_key][[0]]/255.,batch[clean_key][[0]]/255.
        # noisy = rslice(noisy,region)
        # clean = rslice(clean,region)

        # -- forward --
        gpu_mem.print_peak_gpu_stats(False,"test",reset=True)
        with th.no_grad():
            deno = self.forward(noisy,True)
        mem_res,mem_alloc = gpu_mem.print_peak_gpu_stats(False,"test",reset=True)

        # -- compare --
        loss = th.mean((clean - deno)**2)
        psnr = np.mean(compute_psnrs(deno,clean,div=1.)).item()
        ssim = np.mean(compute_ssims(deno,clean,div=1.)).item()

        # -- terminal log --
        self.log("psnr", psnr, on_step=True, on_epoch=False, batch_size=1)
        self.log("ssim", ssim, on_step=True, on_epoch=False, batch_size=1)
        self.log("index",  int(index.item()),on_step=True,on_epoch=False,batch_size=1)
        self.log("mem_res",  mem_res, on_step=True, on_epoch=False, batch_size=1)
        self.log("mem_alloc",  mem_alloc, on_step=True, on_epoch=False, batch_size=1)
        self.gen_loger.info("te_psnr: %2.2f" % psnr)

        # -- log --
        results = edict()
        results.test_loss = loss.item()
        results.test_psnr = psnr
        results.test_ssim = ssim
        results.test_mem_alloc = mem_alloc
        results.test_mem_res = mem_res
        results.test_index = index.cpu().numpy().item()
        return results

class MetricsCallback(Callback):
    """PyTorch Lightning metric callback."""

    def __init__(self):
        super().__init__()
        self.metrics = {}

    def _accumulate_results(self,each_me):
        for key,val in each_me.items():
            if not(key in self.metrics):
                self.metrics[key] = []
            if hasattr(val,"ndim"):
                ndim = val.ndim
                val = val.cpu().numpy().item()
            self.metrics[key].append(val)

    @rank_zero_only
    def log_metrics(self, metrics, step):
        # metrics is a dictionary of metric names and values
        # your code to record metrics goes here
        print("logging metrics: ",metrics,step)

    def on_train_epoch_end(self, trainer, pl_module):
        each_me = copy.deepcopy(trainer.callback_metrics)
        self._accumulate_results(each_me)

    def on_validation_epoch_end(self, trainer, pl_module):
        each_me = copy.deepcopy(trainer.callback_metrics)
        self._accumulate_results(each_me)

    def on_test_epoch_end(self, trainer, pl_module):
        each_me = copy.deepcopy(trainer.callback_metrics)
        self._accumulate_results(each_me)

    def on_train_batch_end(self, trainer, pl_module, outs,
                           batch, batch_idx, dl_idx):
        each_me = copy.deepcopy(trainer.callback_metrics)
        self._accumulate_results(each_me)


    def on_validation_batch_end(self, trainer, pl_module, outs,
                                batch, batch_idx, dl_idx):
        each_me = copy.deepcopy(trainer.callback_metrics)
        self._accumulate_results(each_me)

    def on_test_batch_end(self, trainer, pl_module, outs,
                          batch, batch_idx, dl_idx):
        self._accumulate_results(outs)


def remove_lightning_load_state(state):
    names = list(state.keys())
    for name in names:
        name_new = name.split(".")[1:]
        name_new = ".".join(name_new)
        state[name_new] = state[name]
        del state[name]
