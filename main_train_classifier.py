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
from data_loader import DataLoaderClassification
import cv2

def get_args_parser():
    parser = argparse.ArgumentParser('Training the binary classifier', add_help=False)
    parser.add_argument('--num_images', default=10, type=int)
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--train_hr_dir', type=str, help='hq location')
    parser.add_argument('--train_ttt_dir', type=str, help='ttt location')
    parser.add_argument('--train_pretrain_dir', type=str, help='pretrain location')
    parser.add_argument('--test_hr_dir', type=str, help='hq location')
    parser.add_argument('--test_ttt_dir', type=str, help='ttt location')
    parser.add_argument('--test_pretrain_dir', type=str, help='pretrain location')
    parser.add_argument('--output_dir', type=str, help="location for new model checkpoints")
    parser.add_argument('--threshold', type=float, help="The threshold for the model.")
    parser.add_argument('--batch_size', type=int, default=128, help="Batch size")
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--seed', type=int, help='training seed')
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)
    return parser

def main(args):
    # use cuda if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # instantiate model
    model = net(upscale=1, in_chans=6, out_chans=1, img_size=48, window_size=8,
                img_range=1., depths=[6, 6, 6, 6, 6, 6], embed_dim=180, num_heads=[6, 6, 6, 6, 6, 6],
                mlp_ratio=2, upsampler='pixelshuffle', resi_connection='1conv', rescale_back=False)
    optimizer = torch.optim.Adam(model.parameters())
    param_key_g = 'params'
    model = model.to(device)
    optimizer_to(optimizer, device)

    criterion = torch.nn.BCEWithLogitsLoss(reduction='none').cuda()

    # set random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    save_dir = args.output_dir

    # TTT checkpoint loop for each test image
    train_dataset = DataLoaderClassification(args.train_hr_dir, args.train_pretrain_dir,    
                                             args.train_ttt_dir, threshold=args.threshold)
    test_dataset = DataLoaderClassification(args.test_hr_dir, args.test_pretest_dir, args.test_ttt_dir, 
                                            threshold=args.threshold)
    
    sampler_train = torch.utils.data.RandomSampler(dataset_train)
    sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    train_data_loader = torch.utils.data.DataLoader(train_dataset, 
        sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,)

    test_data_loader = torch.utils.data.DataLoader(
        test_dataset, sampler=sampler_val,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False
    )

    for epoch in range(args.epochs):
        adjust_learning_rate(optimizer, epoch)
        prec1 = train(args, train_data_loader, model, optimizer, criterion, epoch, device)
        test_prec1 = test(args, test_data_loader, model, criterion, epoch, device)
        best_prec1 = max(prec1, best_prec1)
        state_dict = {
            'epoch': epoch + 1,
            'arch': 'swinir',
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
            'test_prec1': test_prec1
        }
        if (epoch) % 10 == 0:
            torch.save(state_dict['state_dict'], f'{save_dir}/checkpoint_swinir_{epoch}.pth')


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


@torch.no_grad()
def test(args, data_loader, model, criterion, epoch, device):
    model.eval()
    losses = AverageMeter()
    acccuracies = AverageMeter()
    for idx, (pretrain_input, ttt_input, mask, signal_mask) in enumerate(data_loader):
        # compute output
        pretrain_input = pretrain_input.to(device, non_blocking=True)
        ttt_input = ttt_input.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)
        signal_mask = signal_mask.to(device, non_blocking=True)
        
        output = model(torch.cat([pretrain_input, ttt_input], axis=1))
        loss = criterion(output, mask)
        loss = torch.sum(loss * signal_mask) / torch.sum(signal_mask)
        # record loss
        accuracy = torch.sum(((output >= 0) ==  mask) * signal_mask) / torch.sum(signal_mask)
        acccuracies.update(accuracy.data.item(), 1)
        losses.update(loss.data.item(), input.size(0))

        if idx % 5 == 0:
            print('Epoch: [{0}][{1}]\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t Acc {acc.val:.4f} ({acc.avg:.4f})'.format(
                   epoch, idx, loss=losses, acc=acccuracies))


def train(args, data_loader, model, optimizer, criterion, epoch, device):
    losses = AverageMeter()
    acccuracies = AverageMeter()
    model.train()
    for idx, (pretrain_input, ttt_input, mask, signal_mask) in enumerate(data_loader):
        # compute output
        pretrain_input = pretrain_input.to(device, non_blocking=True)
        ttt_input = ttt_input.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)
        signal_mask = signal_mask.to(device, non_blocking=True)
        
        output = model(torch.cat([pretrain_input, ttt_input], axis=1))
        loss = criterion(output, mask)
        # record loss
        losses.update(loss.data.item(), input.size(0))
        with torch.no_grad():
            accuracy = torch.sum(((output >= 0) ==  mask) * signal_mask) / torch.sum(signal_mask)
            acccuracies.update(accuracy.data.item(), 1)
        
        # compute gradient and do optimizer step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if idx % 5 == 0:
            print('Epoch: [{0}][{1}]\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\tAcc {acc.val:.4f} ({acc.avg:.4f})'.format(
                   epoch, idx, loss=losses, acc=acccuracies))


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
