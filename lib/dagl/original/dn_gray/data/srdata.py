import os
import glob
import random
import pickle

from data import common

import numpy as np
import imageio
import torch
import torch.utils.data as data
import cv2


class SRData(data.Dataset):
    def __init__(self, args, name='', train=True, benchmark=False):
        self.args = args
        self.name = name
        self.train = train
        self.split = 'train' if train else 'test'
        self.do_eval = True
        self.benchmark = benchmark
        self.input_large = (args.model == 'VDSR')
        self.scale = args.scale
        self.idx_scale = 0

        self._set_filesystem(args.dir_data)
        if args.ext.find('img') < 0:
            path_bin = os.path.join(self.apath, 'bin')
            os.makedirs(path_bin, exist_ok=True)

        self.images_hr = self._scan()
        if train:
            n_patches = args.batch_size * args.test_every#*2# need remove 2 when batch_size=32
            n_images = len(args.data_train) * len(self.images_hr)
            if n_images == 0:
                self.repeat = 0
            else:
                self.repeat = max(n_patches // n_images, 1)#//2

    # Below functions as used to prepare images
    def _scan(self):
        name_hr = []
        list_hr = os.listdir(self.dir_hr)
        for i in list_hr:
            if i.endswith('.png') or i.endswith('jpeg'):
                name_hr.append(os.path.join(self.dir_hr,i))
        name_hr.sort()
        return name_hr

    def _set_filesystem(self, dir_data):
        self.apath = os.path.join(dir_data, self.name)
        self.dir_hr = os.path.join(self.apath, 'HR')
        self.dir_lr = os.path.join(self.apath, 'LR_bicubic')
        if self.input_large: self.dir_lr += 'L'
        self.ext = '.png'

    def _check_and_load(self, ext, img, f, verbose=True):
        if not os.path.isfile(f) or ext.find('reset') >= 0:
            if verbose:
                print('Making a binary: {}'.format(f))
            with open(f, 'wb') as _f:
                pickle.dump(imageio.imread(img), _f)

    def __getitem__(self, idx):
        hr, filename = self._load_file(idx)
        hr = self.get_patch(hr)

        hr = common.set_channel([hr], self.args.n_colors)[0]
        hr_tensor = common.np2Tensor([hr], self.args.rgb_range)[0]
        return hr_tensor, filename

    def __len__(self):
        if self.train:                                 # method of repeat
            return len(self.images_hr) * self.repeat
        else:
            return len(self.images_hr)

    def _get_index(self, idx):
        if self.train:                               # method of repeat
            return idx % len(self.images_hr)
        else:
            return idx

    def _load_file(self, idx):
        idx = self._get_index(idx)
        f_hr = self.images_hr[idx]

        filename, _ = os.path.splitext(os.path.basename(f_hr))
        if self.args.ext == 'img' or self.benchmark:
            hr = cv2.imread(f_hr)[:,:,0]

        return hr, filename

    def get_patch(self, hr):
        patch_size = self.args.patch_size
        scale = self.scale[self.idx_scale]
        multi_scale = len(self.scale) > 1
        if self.train:
            hr = common.get_patch(
                hr, patch_size, scale, multi_scale=multi_scale
            )
            hr = common.augment([hr])[0]

        return hr

    def set_scale(self, idx_scale):
        if not self.input_large:
            self.idx_scale = idx_scale
        else:
            self.idx_scale = random.randint(0, len(self.scale) - 1)
