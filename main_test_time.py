import argparse
import imp
import cv2
import glob
import numpy as np
from collections import OrderedDict
import os
import torch
import requests
from imresize import imresize
from models.network_swinir import SwinIR as net
from utils import utils_image as util
import os.path
import math
import argparse
import time
import random
import numpy as np
from collections import OrderedDict
import logging
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch
import matplotlib.pyplot as plt

from utils import utils_logger
from utils import utils_option as option
from data_loader import DataLoaderPretrained

def main():
    # Pretrained Model
    pretrained = True
    if pretrained:
        best_prec1 = 0
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # set up model
        model_path = '/home/zeeshan/image-sr/superresolution/swinir_sr_classical_patch48_x2/models/260000_G.pth'
        optimizer_path = '/home/zeeshan/image-sr/superresolution/swinir_sr_classical_patch48_x2/models/260000_optimizerG.pth'
        model = net(upscale=2, in_chans=3, img_size=48, window_size=8,
                    img_range=1., depths=[6, 6, 6, 6, 6, 6], embed_dim=180, num_heads=[6, 6, 6, 6, 6, 6],
                    mlp_ratio=2, upsampler='pixelshuffle', resi_connection='1conv')
        optimizer = torch.optim.Adam(model.parameters())
        param_key_g = 'params'
        model = model.to(device)
        optimizer_to(optimizer, device)
        pretrained_model = torch.load(model_path)
        pretrained_optimizer = torch.load(optimizer_path, map_location='cpu')
        model.load_state_dict(pretrained_model[param_key_g] if param_key_g in pretrained_model.keys() else pretrained_model, strict=True)
        optimizer.load_state_dict(pretrained_optimizer[param_key_g] if param_key_g in pretrained_optimizer.keys() else pretrained_optimizer)
        criterion = torch.nn.L1Loss().cuda()
        # Check if model/optimizer loaded correctly
        # print(model)
        # print(optimizer)
        seed = random.randint(1, 10000)
        print('Random seed: {}'.format(seed))
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        save_dir = f'results/swinir_sr_classical_test_time__x2'
        folder = '/home/zeeshan/image-sr/testsets/Set14/LRbicx2'
        border = 2
        window_size = 8
        for idx, path in enumerate(sorted(glob.glob(os.path.join(folder, '*')))):
            print(path)
            img_lq = cv2.imread(path, cv2.IMREAD_COLOR).astype(np.float32) / 255
            data_loader = DataLoaderPretrained(img_lq, sf=2)
            for epoch in range(25):
                adjust_learning_rate(optimizer, epoch)
                prec1 = train(data_loader, model, optimizer, criterion, epoch, device)
                best_prec1 = max(prec1, best_prec1)
                state_dict = {
                    'epoch': epoch + 1,
                    'arch': 'swinir',
                    'state_dict': model.state_dict(),
                    'best_prec1': best_prec1,
                }
                img_name = path[45:].replace('.png', '')
                img_name = img_name.replace('/', '')
                if epoch % 2 == 0:
                    torch.save(state_dict['state_dict'], f'/home/zeeshan/image-sr/test_time_training/checkpoints/x2/checkpoint_swinir_{img_name}_{epoch}.pth')
            model.load_state_dict(pretrained_model[param_key_g] if param_key_g in pretrained_model.keys() else pretrained_model, strict=True)
    
def optimizer_to(optim, device):
    for param in optim.state.values():
        # Not sure there are any global tensors in the state dict
        if isinstance(param, torch.Tensor):
            param.data = param.data.to(device)
            if param._grad is not None:
                param._grad.data = param._grad.data.to(device)
        elif isinstance(param, dict):
            for subparam in param.values():
                if isinstance(subparam, torch.Tensor):
                    subparam.data = subparam.data.to(device)
                    if subparam._grad is not None:
                        subparam._grad.data = subparam._grad.data.to(device)

def train(data_loader, model, optimizer, criterion, epoch, device):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    end = time.time()
    data_loader.generate_pairs(100)
    data = zip(data_loader.lr, data_loader.hr)
    for idx, (input, target) in enumerate(data):
        # measure data loading time
        model.train()
        data_time.update(time.time() - end)
        
        input = torch.from_numpy(input).float().to(device)
        target = torch.from_numpy(target).float().to(device)
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)

        # compute output
        # print("Input:", input.shape)
        output = model(input_var)
        # print("Input:", input.shape, "Output:", output.shape, "Target:", target.shape)
        # output = output[:, :, :128, :128]
        loss = criterion(output, target_var)

        # record loss
        losses.update(loss.data.item(), input.size(0))

        # compute gradient and do optimizer step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if idx % 5 == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                   epoch, idx, len(data_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses))
    return top1.avg

def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = 2e-4 * (0.1 ** (epoch // 30))
    for param_group in optimizer.state_dict()['param_groups']:
        param_group['lr'] = lr

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


if __name__ == '__main__':
    main()

