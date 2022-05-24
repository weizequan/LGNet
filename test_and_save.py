# -*- coding: utf-8 -*-

import cv2

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter

from PIL import Image
import numpy as np
import math
import time
import os
import datetime
import random
import shutil
from options.test_options import TestOptions
from models import create_model
import torchvision.transforms.functional as F
import torch

def calculate_psnr(img1, img2, max_value=255):
    """"Calculating peak signal-to-noise ratio (PSNR) between two images."""
    mse = np.mean((np.array(img1, dtype=np.float32) - np.array(img2, dtype=np.float32)) ** 2)
    if mse == 0:
        return 100
    return 20 * np.log10(max_value / (np.sqrt(mse)))

import glob
def load_flist(flist):
    # np.genfromtxt(flist, dtype=np.str, encoding='utf-8')
    if isinstance(flist, list):
        return flist
    # flist: image file path, image directory path, text file flist path
    if isinstance(flist, str):
        if os.path.isdir(flist):
            flist = list(glob.glob(flist + '/*.jpg')) + list(glob.glob(flist + '/*.png'))
            flist.sort()
            return flist

        if os.path.isfile(flist):
            try:
                return np.genfromtxt(flist, dtype=np.str, encoding='utf-8')
            except:
                return [flist]
    return []

def postprocess(img):
  img = (img+1)/2*255
  img = img.permute(0,2,3,1)
  img = img.int().cpu().numpy().astype(np.uint8)
  return img
  
# load test data
val_image = '/home/wzquan/celeba_HQ/HQ_val'

# Model and version
opt = TestOptions().parse()  # get test options
# hard-code some parameters for test
opt.num_threads = 0   # test code only supports num_threads = 1
opt.batch_size = 1    # test code only supports batch_size = 1
opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
opt.no_flip = True    # no flip; comment this line if results on flipped images are needed.
opt.display_id = -1   # no visdom display; the test code saves the results to a HTML file.
model = create_model(opt)      # create a model given opt.model and other options
model.setup(opt)               # regular setup: load and print networks; create schedulers

model.eval()

val_mask_suffix = ['mask0-10', 'mask10-20', 'mask20-30', 'mask30-40', 'mask40-50', 'mask50-60']
save_dir_suffix = ['010', '1020', '2030', '3040', '4050', '5060']

for suffix_idx in range(6):

    val_mask = '/home/wzquan/mask/test_mask_256/' + val_mask_suffix[suffix_idx]

    save_dir = './results/CelebaHQ/LGNet-' + save_dir_suffix[suffix_idx]

    test_image_flist = load_flist(val_image)
    print(len(test_image_flist))
    test_mask_flist = load_flist(val_mask)
    print(len(test_mask_flist))

    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)

    os.makedirs(os.path.join(save_dir, 'comp'), exist_ok=True)
    os.makedirs(os.path.join(save_dir, 'masked'), exist_ok=True)

    psnr = []
    mask_num = len(test_mask_flist)
    # iteration through datasets
    for idx in range(len(test_image_flist)):
        img = Image.open(test_image_flist[idx]).convert('RGB')
        mask = Image.open(test_mask_flist[idx%mask_num]).convert('L')

        masks = F.to_tensor(mask)
        images = F.to_tensor(img)*2-1.
        images = images.unsqueeze(0)
        masks = masks.unsqueeze(0)

        data = {'A': images, 'B': masks, 'A_paths': ''}
        model.set_input(data)
        with torch.no_grad():
            model.forward()

        orig_imgs = postprocess(model.images)
        mask_imgs = postprocess(model.masked_images1)
        comp_imgs = postprocess(model.merged_images3)

        psnr_tmp = calculate_psnr(comp_imgs, orig_imgs)
        psnr.append(psnr_tmp)
            
        names = test_image_flist[idx].split('/')
        Image.fromarray(comp_imgs[0]).save(save_dir + '/comp/' + names[-1].split('.')[0] + '_comp.png')
        Image.fromarray(mask_imgs[0]).save(save_dir + '/masked/' + names[-1].split('.')[0] + '_mask.png')

    print('Finish in {}'.format(save_dir))
    print('The avg psnr is',np.mean(np.array(psnr)))
