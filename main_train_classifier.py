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
from utils import utils_calculate_psnr_ssim as util
import cv2
import PIL.Image
import tqdm
import glob

def get_args_parser():
    parser = argparse.ArgumentParser('Training the binary classifier', add_help=False)
    parser.add_argument('--num_images', default=10, type=int)
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--train_hr_dir', type=str, help='hq location')
    parser.add_argument('--train_ttt_dir', type=str, help='ttt location')
    parser.add_argument('--train_pretrain_dir', type=str, help='pretrain location')
    parser.add_argument('--test_hr_dir', type=str, help='hq location')
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
    parser.set_defaults(pin_mem=True)
    return parser


def prep_image(path):
    image = PIL.Image.open(path)
    image = np.array(image) / 255.
    image = image.transpose(2, 0, 1)
    image = image[np.newaxis,...]
    return torch.from_numpy(image).float()

@torch.no_grad()
def merge_and_psnr(model, ttt_path, orig_path, gt_path, device, scale=4):
    # TODO(Zeeshan): Verify that the cropping is right (it is not).
    model.eval()
    merged_psnrs = []
    merged_ssims = []
    ttt_psnrs = []
    ttt_ssims = []
    orig_psnrs = []
    orig_ssims = []
    soft_merged_psnrs = []
    soft_merged_ssims = []
    
    for gt_image_path in tqdm.tqdm(glob.glob(os.path.join(gt_path, '*.png'))):
        image_name = os.path.split(gt_image_path)[-1]
        gt_image = prep_image(gt_image_path).to(device)
        ttt_image = prep_image(os.path.join(ttt_path, image_name)).to(device)
        orig_image = prep_image(os.path.join(orig_path, image_name)).to(device)
        model_output = model(torch.cat([orig_image, ttt_image], axis=1)) 
        output = (model_output > 0).float()
        
        merged = ttt_image * output + orig_image * (1 - output)
        soft_merged = ttt_image * torch.sigmoid(model_output) + orig_image * (1 - torch.sigmoid(model_output))
        gt_image = gt_image[:, :, :merged.shape[2], :merged.shape[3]]
        assert merged.shape == gt_image.shape
        gt_image = (gt_image[0].detach().cpu().numpy().transpose(1, 2, 0) * 255.0).round().astype(np.uint8)
        merged = (merged[0].detach().cpu().numpy().transpose(1, 2, 0) * 255.0).round().astype(np.uint8)
        ttt_image = (ttt_image[0].detach().cpu().numpy().transpose(1, 2, 0) * 255.0).round().astype(np.uint8)
        orig_image = (orig_image[0].detach().cpu().numpy().transpose(1, 2, 0) * 255.0).round().astype(np.uint8)
        soft_merged = (soft_merged[0].detach().cpu().numpy().transpose(1, 2, 0) * 255.0).round().astype(np.uint8)
        # Merged:
        psnr_y = util.calculate_psnr(merged, gt_image, crop_border=scale, test_y_channel=True)
        ssim_y = util.calculate_ssim(merged, gt_image, crop_border=scale, test_y_channel=True)
        merged_psnrs.append(psnr_y)
        merged_ssims.append(ssim_y)
        # TTT
        psnr_y = util.calculate_psnr(ttt_image, gt_image, crop_border=scale, test_y_channel=True)
        ssim_y = util.calculate_ssim(ttt_image, gt_image, crop_border=scale, test_y_channel=True)
        ttt_psnrs.append(psnr_y)
        ttt_ssims.append(ssim_y)
        # ORIGINAL
        psnr_y = util.calculate_psnr(orig_image, gt_image, crop_border=scale, test_y_channel=True)
        ssim_y = util.calculate_ssim(orig_image, gt_image, crop_border=scale, test_y_channel=True)
        orig_psnrs.append(psnr_y)
        orig_ssims.append(ssim_y)
        # Soft merge
        psnr_y = util.calculate_psnr(soft_merged, gt_image, crop_border=scale, test_y_channel=True)
        ssim_y = util.calculate_ssim(soft_merged, gt_image, crop_border=scale, test_y_channel=True)
        soft_merged_psnrs.append(psnr_y)
        soft_merged_ssims.append(ssim_y)
        
    print('Mean orig psnr:', np.mean(orig_psnrs))
    print('Mean orig ssim:', np.mean(orig_ssims))
    print('Mean ttt psnr:', np.mean(ttt_psnrs))
    print('Mean ttt ssim:', np.mean(ttt_ssims))
    print('Mean merged psnr:', np.mean(merged_psnrs))
    print('Mean merged ssim:', np.mean(merged_ssims))
    print('Soft mean merged psnr:', np.mean(soft_merged_psnrs))
    print('Soft mean merged ssim:', np.mean(soft_merged_ssims))



def main(args):
    # use cuda if available
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    
    # instantiate model
    model = net(upscale=1, in_chans=6, out_chans=1, 
                img_size=args.img_size, window_size=8,
                img_range=1., depths=[6, 6, 6, 6, 6, 6, 6], 
                embed_dim=180, num_heads=[6, 6, 6, 6, 6, 6, 6],
                mlp_ratio=2, upsampler='pixelshuffle', 
                resi_connection='1conv', rescale_back=False)
    # The model is a bit bigger than the original model (one more layer)
    best_prec1 = 0
    
    print('num_params', sum(p.numel() for p in model.parameters() if p.requires_grad))
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    model = model.to(device)
    optimizer_to(optimizer, device)

    criterion = torch.nn.BCEWithLogitsLoss(reduction='none').to(device)

    # set random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    save_dir = args.output_dir
    if not os.path.exists(args.output_dir):
         os.mkdir(args.output_dir)
    print(f'Saving to {save_dir}')
    # TTT checkpoint loop for each test image
    train_dataset = DataLoaderClassification(args.train_hr_dir, args.train_pretrain_dir,    
                                             args.train_ttt_dir, threshold=args.threshold,
                                             split='train',
                                             img_size=args.img_size)
    test_dataset = DataLoaderClassification(args.test_hr_dir, 
                                            args.test_pretrain_dir, 
                                            args.test_ttt_dir, 
                                            threshold=args.threshold, split='test',
                                            img_size=args.img_size)
    
    sampler_train = torch.utils.data.RandomSampler(train_dataset)
    sampler_val = torch.utils.data.SequentialSampler(test_dataset)

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
        if (epoch + 1) % 50 == 0:
            torch.save(state_dict['state_dict'], f'{save_dir}/checkpoint_swinir_{epoch}_{prec1:.2f}_{test_prec1:.2f}.pth')


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
        losses.update(loss.data.item(), pretrain_input.size(0))
        # TODO(yossi): maybe also run the psnr etc. 
        print('Epoch: [{0}][{1}]\t'
                'Loss {loss.val:.4f} ({loss.avg:.4f})\t Acc {acc.val:.4f} ({acc.avg:.4f})'.format(
                epoch, idx, loss=losses, acc=acccuracies))
    return acccuracies.avg


import math

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


def train(args, data_loader, model, optimizer, criterion, epoch, device):
    losses = AverageMeter()
    acccuracies = AverageMeter()
    lr = AverageMeter()
    model.train()
    for idx, (pretrain_input, ttt_input, mask, signal_mask) in enumerate(data_loader):
        # compute output
        adjust_learning_rate(optimizer, idx / len(data_loader) + epoch, args)

        pretrain_input = pretrain_input.to(device, non_blocking=True)
        ttt_input = ttt_input.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)
        signal_mask = signal_mask.to(device, non_blocking=True)
        
        output = model(torch.cat([pretrain_input, ttt_input], axis=1))
        loss = criterion(output, mask)
        assert loss.shape == signal_mask.shape
        loss = torch.sum(loss * signal_mask) / torch.sum(signal_mask)
        # record loss
        losses.update(loss.data.item(), pretrain_input.size(0))
        curr_lr = optimizer.param_groups[0]["lr"]
        lr.update(curr_lr)
        with torch.no_grad():
            accuracy = torch.sum(((output >= 0) ==  mask) * signal_mask) / torch.sum(signal_mask)
            acccuracies.update(accuracy.data.item(), 1)
        
        # compute gradient and do optimizer step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print('Epoch: [{0}][{1}]\t'
                'Loss {loss.val:.4f} ({loss.avg:.4f})\tAcc {acc.val:.4f} ({acc.avg:.4f})\t Lr {lr.val:.4f}'.format(
                epoch, idx, loss=losses, acc=acccuracies, lr=lr))
    return acccuracies.avg


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
