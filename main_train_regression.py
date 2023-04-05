import os
from pathlib import Path
import argparse
import sys
import glob
import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from models.network_swinir import SwinIR as net
import os.path
import time
import random
from data_loader import DataLoaderRegression
from utils import utils_calculate_psnr_ssim as util
import cv2
import PIL.Image
import tqdm
import glob
import math

def get_args_parser():
    parser = argparse.ArgumentParser('Training the binary classifier', add_help=False)
    parser.add_argument('--num_images', default=10, type=int)
    parser.add_argument('--epochs', default=10, type=int)
    parser.add_argument('--train_hr_dir', type=str, help='hq location')
    parser.add_argument('--train_lr_dir', type=str, help='lr location')
    parser.add_argument('--train_ttt_dir', type=str, help='ttt location')
    parser.add_argument('--train_pretrain_dir', type=str, help='pretrain location')
    parser.add_argument('--test_hr_dir', type=str, help='hq location')
    parser.add_argument('--test_lr_dir', type=str, help='lr location')
    parser.add_argument('--test_ttt_dir', type=str, help='ttt location')
    parser.add_argument('--test_pretrain_dir', type=str, help='pretrain location')
    parser.add_argument('--output_dir', type=str, help="location for new model checkpoints")
    parser.add_argument('--threshold', type=float, help="The threshold for the model.")
    parser.add_argument('--batch_size', type=int, default=32, help="Batch size")
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--lr', default=0.00001, type=float, help='Learning rate')
    parser.add_argument('--min_lr', default=0.000001, type=float, help='Min learning rate')
    parser.add_argument('--seed', default=2023, type=int, help='training seed')
    parser.add_argument('--img_size', default=48, type=int, help='training seed')
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.add_argument('--warmup_epochs', type=int, default=10, metavar='N',
                        help='epochs to warmup LR')
    parser.add_argument('--window_size', type=int, default=8, help='SwinIR model window size')
    parser.add_argument('--opt_type', type=str, default='Adam', help='Optimizer for SwinIR model (SGD/Adam)')
    parser.set_defaults(pin_mem=True)
    return parser

def train_worker(args):
    # Use CUDA if available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Instantiate model
    model = net(upscale=1, in_chans=6, out_chans=1, 
                img_size=args.img_size, window_size=args.window_size,
                img_range=1., depths=[6, 6, 6, 6, 6, 6, 6], 
                embed_dim=180, num_heads=[6, 6, 6, 6, 6, 6, 6],
                mlp_ratio=2, upsampler='pixelshuffle', 
                resi_connection='1conv', rescale_back=False).to(device)

    # Wrap model with DataParallel
    model = torch.nn.DataParallel(model)

    # Instantiate loss function and optimizer
    if args.opt_type == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    elif args.opt_type == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
    optimizer_to(optimizer, device)
    criterion = torch.nn.MSELoss().to(device)

    # Set random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    save_dir = args.output_dir
    if not os.path.exists(args.output_dir):
         os.mkdir(args.output_dir)
    print(f'Saving to {save_dir}')

    train_dataset = DataLoaderRegression(args.train_hr_dir, args.train_lr_dir, 
                                         args.train_pretrain_dir, args.train_ttt_dir,
                                         split='train', img_size=args.img_size)
    
    test_dataset = DataLoaderRegression(args.test_hr_dir, args.test_lr_dir, 
                                         args.test_pretrain_dir, args.test_ttt_dir,
                                         split='test', img_size=args.img_size)
    sampler_train = torch.utils.data.RandomSampler(train_dataset)
    sampler_val = torch.utils.data.SequentialSampler(test_dataset)

    train_data_loader = torch.utils.data.DataLoader(train_dataset, 
        sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True
    )

    test_data_loader = torch.utils.data.DataLoader(
        test_dataset, sampler=sampler_val,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False
    )

    # Training loop
    print("Training Batches:", len(train_data_loader))
    best_prec1 = 0
    for epoch in range(args.epochs):
        print(f"Starting epoch {epoch}...")
        prec1 = train(args, train_data_loader, model, optimizer, criterion, epoch, device)
        print(f'Done training. Epoch [{epoch:05}]: {prec1}')
        test_prec1 = test(args, test_data_loader, model, criterion, epoch, device)
        best_prec1 = max(prec1, best_prec1)
        state_dict = {
            'epoch': epoch + 1,
            'arch': 'swinir',
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
            'test_prec1': test_prec1
        }
        print(f'Done testing. Epoch [{epoch:05}]: {test_prec1}')
        if (epoch + 1) % 2 == 0:
            torch.save(state_dict['state_dict'], f'{save_dir}/checkpoint_swinir_s{args.img_size}_ep{epoch}_win{args.window_size}_{prec1:.2f}_{test_prec1:.2f}.pth')

# TODO: Write training and testing functions
def train():
    pass

def test():
    pass




def adjust_learning_rate(optimizer, epoch, args):
    """Decay the learning rate with half-cycle cosine after warmup"""
    if epoch < args.warmup_epochs:
        lr = args.lr * epoch / args.warmup_epochs 
    else:
        lr = args.min_lr + (args.lr - args.min_lr) * 0.5 * \
            (1. + math.cos(math.pi * (epoch - args.warmup_epochs) / (args.epochs - args.warmup_epochs)))
    for param_group in optimizer.param_groups:
        if "lr_scale" in param_group:
            param_group["lr"] = lr * param_group["lr_scale"]
        else:
            param_group["lr"] = lr
    return lr


def optimizer_to(optim, device):
    """
    Move pretrained optimizer to device used by PyTorch
    """
    for param in optim.state.values():
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