import torch, torchvision
import torch.nn
import numpy as np
import PIL.Image
from models.network_swinir import SwinIR as net
from utils import utils_calculate_psnr_ssim as util
import os
import glob
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
import torch.distributed as dist
from collections import OrderedDict
import csv
import argparse
import pandas as pd
import cv2
import re
from io import BytesIO

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb=128'
class SwinIR_Merger:
    def __init__(self, pretrained_model, ttt_test_checkpoints, test_images, alpha, quality, device):
        self.pretrained_model = pretrained_model
        self.ttt_test_checkpoints = ttt_test_checkpoints
        self.test_images = test_images
        self.alpha = alpha
        self.quality = quality
        self.device = device
    
    def degrade_img(self, img):
        original_img = PIL.Image.open(img)
        buffer = BytesIO()
        original_img.save(buffer, format='JPEG', quality=self.quality)
        buffer.seek(0)
        compressed_image = PIL.Image.open(buffer)
        to_tensor = torchvision.transforms.ToTensor()
        compressed_image = to_tensor(compressed_image).to(self.device)
        compressed_image = compressed_image[None, :]
        original_img = np.array(original_img)
        return compressed_image, original_img

    def merge_weights(self, model1, model2):
        merged_model = OrderedDict()
        for key in model1:
            if key.startswith('module.'):
                key_without_module = key[len('module.'):]
                if key_without_module in model2:
                    merged_model[key] = (1 - self.alpha) * model1[key] + self.alpha * model2[key_without_module]
                else:
                    merged_model[key] = model1[key]
            else:
                if key in model2:
                    merged_model[key] = (1 - self.alpha) * model1[key] + self.alpha * model2[key]
                else:
                    merged_model[key] = model1[key]
        return merged_model

    def inference(self, input, model):
        with torch.no_grad():
            output = model(input).to(self.device)
            output = output.data.squeeze().float().clamp_(0, 1).cpu().numpy()
            if output.ndim == 3:
                output = np.transpose(output, (1, 2, 0))
            return np.uint8((output*255.0).round())
    
    def pad_to_multiple_of_7(self, img):
        h, w = img.shape[1:]
        h_pad = 7 - h % 7 if h % 7 != 0 else 0
        w_pad = 7 - w % 7 if w % 7 != 0 else 0
        img_padded = np.pad(img, ((0, h_pad), (0, w_pad), (0, 0)), mode='constant')
        return img_padded

    def merge_and_infer(self, args):
        csv_filename = f"/home/zeeshan/image-sr/jpeg_merge/classic5_test_mse_ttt_q{self.quality}_{args.epochs}.csv"
        ttt_model = net(upscale=1, in_chans=1, out_chans=1, img_size=126, window_size=7, img_range=255.0, 
                    depths=[6, 6, 6, 6, 6, 6], embed_dim=180, num_heads=[6, 6, 6, 6, 6, 6],
                    mlp_ratio=2, upsampler=None, resi_connection='1conv')

        print("Merging Testing Checkpoints...")
        # alphas = [0.01, 0.02, 0.03, 0.04, 0.045, 0.0475, 0.05, 0.051, 0.0525, 0.055, 0.5, 1.]
        alphas = [round(0.01 + 0.01*i, 2) for i in range(0, 10)] + [1.]
        for idx, (ttt_checkpoint, test_image) in enumerate(zip(self.ttt_test_checkpoints, self.test_images)):
            compressed_img, original_img = self.degrade_img(test_image)
            original_img = original_img[np.newaxis, ...]
            original_img_padded = self.pad_to_multiple_of_7(original_img)
            best_psnr, best_ssim, best_psnr_b, best_alpha = 0, 0, 0, 0
            for alpha in alphas:
                self.alpha = alpha
                ttt_model_weights = torch.load(ttt_checkpoint, map_location=self.device)
                ttt_model.load_state_dict(ttt_model_weights)
                ttt_model.to('cpu')
                self.pretrained_model.to('cpu')
                merged_weights = self.merge_weights(self.pretrained_model.state_dict(), ttt_model.state_dict())

                merged_model = self.pretrained_model
                merged_model.load_state_dict(merged_weights)
                merged_model.to(self.device)
                merged_model.eval()
                output_image = self.inference(compressed_img, merged_model)
                torch.cuda.empty_cache()
                output_image = output_image[np.newaxis, ...]
                psnr_y = util.calculate_psnr(output_image, original_img, crop_border=0, test_y_channel=True, input_order='CHW')
                ssim_y = util.calculate_ssim(output_image, original_img, crop_border=0, test_y_channel=True, input_order='CHW')
                print(f'Checkpoint: {os.path.basename(ttt_checkpoint)}, Test Image: {os.path.basename(test_image)}, Alpha: {self.alpha}, PSNR: {psnr_y}, SSIM: {ssim_y}')

                # pad the images to match window size
                output_image_padded = self.pad_to_multiple_of_7(output_image)
                psnr_b = util.calculate_psnrb(output_image_padded, original_img_padded, crop_border=0, test_y_channel=False, input_order='CHW')
                if psnr_y > best_psnr:
                    best_psnr, best_ssim, best_psnr_b, best_alpha = psnr_y, ssim_y, psnr_b, alpha
                torch.cuda.empty_cache()
            with open(csv_filename, mode='a') as csv_file:
                csv_writer = csv.writer(csv_file)
                if csv_file.tell() == 0:
                    csv_writer.writerow(['Image Path', 'Alpha', 'PSNR', 'PSNR_B', 'SSIM', 'Type'])
                csv_writer.writerow([test_image, best_alpha, best_psnr, best_psnr_b, best_ssim, 'test'])

def main_worker(local_rank, world_size, args):
    # DDP set up
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = args.port
    dist.init_process_group(backend='nccl', rank=local_rank, world_size=world_size)

    # Set relevant directories
    TEST_IMG_DIR='/home/zeeshan/image-sr/testsets/classic5/'
    TEST_TTT_DIR=f'/old_home_that_will_be_deleted_at_some_point/zeeshan/classic5/ttt_mse_{args.epochs}_q{args.quality}/'
    PRETRAINED_MODEL_PATH = f'/home/zeeshan/image-sr/jpeg_compression/006_CAR_DFWB_s126w7_SwinIR-M_jpeg{args.quality}.pth'

    # Get all file paths
    ttt_test_checkpoints = sorted(glob.glob(os.path.join(TEST_TTT_DIR, '*.pth')))[local_rank::world_size]
    test_images = sorted(glob.glob(os.path.join(TEST_IMG_DIR, '*.bmp')))[local_rank::world_size]

    print("Loaded file paths!")

    # Instantiate pretrained model 
    device = torch.device('cuda', local_rank)
    model = net(upscale=1, in_chans=1, out_chans=1, img_size=126, window_size=7, 
                img_range=255.0, depths=[6, 6, 6, 6, 6, 6], embed_dim=180, 
                num_heads=[6, 6, 6, 6, 6, 6],mlp_ratio=2, upsampler=None, resi_connection='1conv')
    pretrained_model = torch.load(PRETRAINED_MODEL_PATH, map_location=device)
    param_key_g = 'params'
    model.load_state_dict(pretrained_model[param_key_g] if param_key_g in pretrained_model.keys() else pretrained_model, strict=True)
    print("Loaded pretrained SwinIR model!")
    model.to(device)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[device], output_device=device)

    # Instantiate merger instance
    alpha, quality = args.alpha, args.quality
    merger = SwinIR_Merger(model, ttt_test_checkpoints, test_images, alpha, quality, device)

    # Run merger
    print('Starting Merge...')
    merger.merge_and_infer(args)
    print('\nCompleted merge!')

def get_args_parser():
    parser = argparse.ArgumentParser('Model weighting', add_help=False)
    parser.add_argument('--alpha', default=0.1, type=float)
    parser.add_argument('--quality', default=10, type=int)
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--port', type=str)
    return parser

def main():
    parser = get_args_parser()
    args = parser.parse_args()
    world_size = torch.cuda.device_count()
    mp.spawn(main_worker, args=(world_size, args), nprocs=world_size, join=True)

if __name__ == '__main__':
    main()