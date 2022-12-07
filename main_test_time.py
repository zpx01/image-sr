import os
from pathlib import Path
import argparse
import sys
import glob
import numpy as np
import torch
from models.network_swinir import SwinIR as net
import os.path
import time
import random
from data_loader import DataLoaderPretrained
import cv2

def get_args_parser():
    parser = argparse.ArgumentParser('Test Time Training - Classical SR', add_help=False)
    parser.add_argument('--model_path', type=str)
    parser.add_argument('--opt_path', type=str)
    parser.add_argument('--scale', default=4, type=int)
    parser.add_argument('--num_images', default=10, type=int)
    parser.add_argument('--epochs', default=5, type=int)
    parser.add_argument('--test_dir', type=str, help='testset location') 
    parser.add_argument('--output_dir', type=str, help="location for new model checkpoints")
    parser.add_argument('--reset', type=bool, default=True, help='set to False if you do not want to reset optimizer')
    return parser

def main(args):
    # use cuda if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_path = args.model_path
    optimizer_path = args.opt_path

    # instantiate model
    model = net(upscale=args.scale, in_chans=3, img_size=48, window_size=8,
                img_range=1., depths=[6, 6, 6, 6, 6, 6], embed_dim=180, num_heads=[6, 6, 6, 6, 6, 6],
                mlp_ratio=2, upsampler='pixelshuffle', resi_connection='1conv')
    optimizer = torch.optim.Adam(model.parameters())
    param_key_g = 'params'
    model = model.to(device)
    optimizer_to(optimizer, device)

    # load pretrained model
    pretrained_model = torch.load(model_path)
    pretrained_optimizer = torch.load(optimizer_path, map_location='cpu')
    model.load_state_dict(pretrained_model[param_key_g] if param_key_g in pretrained_model.keys() else pretrained_model, strict=True)
    optimizer.load_state_dict(pretrained_optimizer[param_key_g] if param_key_g in pretrained_optimizer.keys() else pretrained_optimizer)
    criterion = torch.nn.L1Loss().cuda()

    # set random seed
    seed = random.randint(1, 10000)
    print('Random seed: {}'.format(seed))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    save_dir = args.output_dir
    folder = args.test_dir
    best_prec1 = 0

    # TTT checkpoint loop for each test image
    for idx, path in enumerate(sorted(glob.glob(os.path.join(folder, '*')))):
        print(path)
        img_lq = cv2.imread(path, cv2.IMREAD_COLOR).astype(np.float32) / 255
        data_loader = DataLoaderPretrained(img_lq, sf=args.scale)
        for epoch in range(args.epochs):
            adjust_learning_rate(optimizer, epoch)
            prec1 = train(args, data_loader, model, optimizer, criterion, epoch, device)
            best_prec1 = max(prec1, best_prec1)
            state_dict = {
                'epoch': epoch + 1,
                'arch': 'swinir',
                'state_dict': model.state_dict(),
                'best_prec1': best_prec1,
            }
            img_name = path[45:].replace('.png', '')
            img_name = img_name.replace('/', '')
            if (epoch) % 2 == 0:
                torch.save(state_dict['state_dict'], f'{save_dir}/checkpoint_swinir_{img_name}_{epoch}.pth')
        model.load_state_dict(pretrained_model[param_key_g] if param_key_g in pretrained_model.keys() else pretrained_model, strict=True)
        if args.reset:
            optimizer.load_state_dict(pretrained_optimizer[param_key_g] if param_key_g in pretrained_optimizer.keys() else pretrained_optimizer)

    

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

def train(args, data_loader, model, optimizer, criterion, epoch, device):
    """
    TTT training, updates pretrained model with new augmentations 
    of test image
    """
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    end = time.time()
    data_loader.generate_pairs(args.num_images)
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
        output = model(input_var)
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
    lr = 2e-5 * (0.1 ** (epoch // 2))
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
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
