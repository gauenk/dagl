import torch
import cv2
import os
import argparse
import glob
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable

import utility
import data
from utils import *
import model as my_model
from option import args

# try:
#     import utility
#     import data
#     from utils import *
#     import model as my_model
#     from option import args
# except:
#     from . import data
#     from . import utility
#     from .utils import *
#     from . import model as my_model
#     from .option import args

import time
import torchvision.utils as vutils
import torch as th

torch.manual_seed(args.seed)
checkpoint = utility.checkpoint(args)
parser = argparse.ArgumentParser(description="GATIR")
parser.add_argument('--logdir',type=str,default='checkpoints/25/model/model_best.pt',help="Path to pretrained model")
parser.add_argument("--test_data", type=str, default='testsets/BSD68', help='test on Set12, BSD68 and Urban100')
parser.add_argument("--test_noiseL", type=float, default=25., help='noise level used on test set')
parser.add_argument("--rgb_range",type=int,default=1.)
parser.add_argument("--save_path",type=str,default='res_vis/GATIR_15',help='Save restoration results')
parser.add_argument("--save",action='store_true')
parser.add_argument("--ensemble",action='store_true')
opt = parser.parse_args()
def normalize(data,rgb_range):
    return data/(255./opt.rgb_range)

# -- helper --
def yuv2rgb(burst):
    # -- weights --
    w = [1./np.sqrt(3),1./np.sqrt(2),np.sqrt(2.)/np.sqrt(3)]
    # -- copy channels --
    y,u,v = burst[...,0].copy(),burst[...,1].copy(),burst[...,2].copy()
    # -- yuv -> rgb --
    burst[...,0] = w[0] * y + w[1] * u + w[2] * 0.5 * v
    burst[...,1] = w[0] * y - w[2] * v
    burst[...,2] = w[0] * y - w[1] * u + w[2] * 0.5 * v
def rgb2yuv(burst):
    # -- weights --
    weights = [1./np.sqrt(3),1./np.sqrt(2),np.sqrt(2.)*2./np.sqrt(3)]
    # -- copy channels --
    r,g,b = burst[...,0].copy(),burst[...,1].copy(),burst[...,2].copy()
    # -- rgb -> yuv --
    burst[...,0] = weights[0] * (r + g + b)
    burst[...,1] = weights[1] * (r - b)
    burst[...,2] = weights[2] * (.25 * r - 0.5 * g + .25 * b)

def main():
    # Build model
    torch.cuda.set_device(0)
    print('Loading model ...\n')
    print(args)
    print("pre.")
    net = my_model.Model(args, checkpoint)
    print("post.")
    print("opt.logdir: ",opt.logdir)
    net.model.load_state_dict(torch.load(opt.logdir,map_location='cuda'))
    print("post2.")
    # this model is trained at cuda1.
    model = net.cuda()
    model.eval()
    opt.ensemble = True

    # load data info
    print('Loading data info ...\n')
    files_source = glob.glob(os.path.join(opt.test_data, '*.png'))
    files_source.sort()
    print(os.path.join(opt.test_data, '*.png'),files_source)
    # process data
    psnr_test = 0
    for f in files_source:
        # image
        Img = cv2.imread(f)

        # Img = np.flip(Img,-1)
        # rgb2yuv(Img)

        # print("Img.shape: ",Img.shape)
        Img = normalize(np.float32(Img[:, :, 0]),opt.rgb_range)
        Img = np.expand_dims(Img, 0)
        Img = np.expand_dims(Img, 1)
        ISource = torch.Tensor(Img)
        # torch.manual_seed(1)    # fixed seed
        print(opt.test_noiseL)
        std_ave = opt.test_noiseL / (255./opt.rgb_range)
        noise = torch.FloatTensor(ISource.size()).normal_(mean=0,std=std_ave)
        # noisy image
        INoisy = ISource + noise
        print(std_ave)
        ISource, INoisy = ISource.cuda(), INoisy.cuda()
        # ISource, INoisy = Variable(ISource.cuda()), Variable(INoisy.cuda())
        with torch.no_grad():  # this can save much memory
            Out = torch.clamp(model(INoisy ,0,ensemble=opt.ensemble), 0., opt.rgb_range)
            print(ISource)
            print(Out)
            image = (Out[0,0].cpu().data.numpy()*(255./opt.rgb_range)).astype(np.uint8)
            # save results
            if opt.save == True:
                if os.path.isdir(opt.save_path) == False:
                    os.mkdir(opt.save_path)
                print(os.path.join(opt.save_path,f.split('/')[-1]))
                image_s = image
                # image_s = np.clip(Img[0,0]*255.,0.,255.)
                print("image.shape: ",image.shape)
                print("Img.shape: ",Img.shape)
                cv2.imwrite(os.path.join(opt.save_path,f.split('/')[-1]),image_s)
        # psnr = -10. * th.log10(th.mean((Out.view(-1) - ISource.view(-1))**2))
        noi_psnr = batch_PSNR(INoisy, ISource, opt.rgb_range)
        psnr = batch_PSNR(Out, ISource, opt.rgb_range)
        psnr_test += psnr
        print("%s Input PSNR %.2f, Output PSNR %.2f" % (f, noi_psnr, psnr))
        exit(0)
    psnr_test /= len(files_source)
    print("\nPSNR on test data %f" % psnr_test)
    print('Finish!')
if __name__ == "__main__":
    main()
