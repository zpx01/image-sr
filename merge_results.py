import os, glob, argparse
import numpy as np
from data_loader_utils import *
import cv2
from collections import defaultdict
import torch
import skimage
from utils import utils_calculate_psnr_ssim as util

res_strs = []
class MergeResults(object):
    """
    MergeResults takes outputs of two classical SR models, one with test-time
    training adaptation and one without. It merges the outputs of the two models
    by taking the best pixels (determined by L2 loss) from both images to form
    a new, merged inference. It provides the PSNR and SSIM for each of the three
    image categories.
    """
    def __init__(self, model_path_1, model_path_2, gt_path, lr_path, merged_img_path, scale):
        self.model_path_1 = model_path_1 # original pretrained model inference
        self.model_path_2 = model_path_2 # ttt model inference 
        self.gt_path = gt_path
        self.lr_path = lr_path
        self.merged_img_path = merged_img_path
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.psnr_values = defaultdict(list)
        self.ssim_values = defaultdict(list)
        self.mse_values = defaultdict(list)
        self.scale = scale
        self.window_size = 8
    
    def merge_results(self):
        # load images
        global res_strs
        img1 = cv2.imread(self.model_path_1, cv2.IMREAD_COLOR)
        img2 = cv2.imread(self.model_path_2, cv2.IMREAD_COLOR)
        gt = cv2.imread(self.gt_path, cv2.IMREAD_COLOR)
        lr = cv2.imread(self.lr_path, cv2.IMREAD_COLOR)
        lr = np.transpose(lr if lr.shape[2] == 1 else lr[:, :, [2, 1, 0]], (2, 0, 1))  # HCW-BGR to CHW-RGB
        lr = torch.from_numpy(lr).float().unsqueeze(0)  # CHW-RGB to NCHW-RGB
        _, _, h_old, w_old = lr.size()
        gt = gt[:h_old * args.scale, :w_old * args.scale, ...]  # crop gt
        gt = np.squeeze(gt)
        (imgname, imgext) = os.path.splitext(os.path.basename(self.gt_path))
        s = f"Testing {imgname} ({self.gt_path})...\n"
        print(s)
        res_strs.append(s)
        # compute individual PSNR/SSIM values
        original_psnr, ttt_psnr = self.psnr(gt, img1, crop_border=self.scale), self.psnr(gt, img2, crop_border=self.scale)
        original_ssim, ttt_ssim = self.ssim(gt, img1, crop_border=self.scale), self.ssim(gt, img2, crop_border=self.scale)

        # create merged image
        merged_img, mse_orig, mse_ttt, mse_merged = self.merge_images(img1, img2, gt)

        # save merged image
        cv2.imwrite(self.merged_img_path, merged_img)
        
        # compute merged PSNR/SSIM values
        merged_psnr = self.psnr(gt, merged_img, crop_border=self.scale)
        merged_ssim = self.ssim(gt, merged_img, crop_border=self.scale)

        # save metrics
        self.psnr_values["original"].append(original_psnr)
        self.psnr_values["ttt"].append(ttt_psnr)
        self.psnr_values["merged"].append(merged_psnr)
        self.ssim_values["original"].append(original_ssim)
        self.ssim_values["ttt"].append(ttt_ssim)
        self.ssim_values["merged"].append(merged_ssim)
        self.mse_values["original"].append(mse_orig)
        self.mse_values["ttt"].append(mse_ttt)
        self.mse_values["merged"].append(mse_merged)

    def psnr(self, img_gt, output, crop_border):
        psnr = util.calculate_psnr(output, img_gt, crop_border=crop_border, test_y_channel=True)
        psnr = float('{:.2f}'.format(psnr))
        return psnr
    
    def ssim(self, img_gt, output, crop_border):
        ssim = util.calculate_ssim(output, img_gt, crop_border=crop_border, test_y_channel=True)
        ssim = float('{:.2f}'.format(ssim))
        return ssim

    def merge_images(self, inf_img1, inf_img2, gt_img):
        inf_img1_y, inf_img2_y, gt_img_y = util.to_y_channel(inf_img1), util.to_y_channel(inf_img2), util.to_y_channel(gt_img)
        loss1 = np.sum((inf_img1_y - gt_img_y) ** 2, axis=-1)
        loss2 = np.sum((inf_img2_y - gt_img_y) ** 2, axis=-1)
        loss1 = loss1[..., np.newaxis]
        loss2 = loss2[..., np.newaxis]
        merged_img = np.where(loss1 < loss2, inf_img1, inf_img2)
        mse_orig = np.mean((inf_img1_y - gt_img_y) ** 2)
        mse_ttt = np.mean((inf_img2_y - gt_img_y) ** 2)
        merged_y = util.to_y_channel(merged_img)
        mse_merged = np.mean((merged_y - gt_img_y) ** 2)
        return merged_img, mse_orig, mse_ttt, mse_merged

def get_args_parser():
    parser = argparse.ArgumentParser('MergeResults - Classical SR', add_help=False)
    parser.add_argument('--pretrained_dir', type=str, help='path to base pretrained model inference images')
    parser.add_argument('--ttt_dir', type=str, help='path to TTT model inference images')
    parser.add_argument('--gt_dir', type=str, help='path to ground truth images')
    parser.add_argument('--merged_dir', type=str, help='path to directory for merged image results')
    parser.add_argument('--lr_dir', type=str, help='path to low res images')
    parser.add_argument('--scale', type=int, default=4)
    parser.add_argument('--results_log', type=str, help='path to text file to save metrics')
    return parser

def main(args):
    global res_strs
    test_img_folder = args.gt_dir
    test_img_paths = []
    for idx, path in enumerate(sorted(glob.glob(os.path.join(test_img_folder, '*')))):
        test_img_paths.append(path)

    model_1_results_dir = args.pretrained_dir
    model_1_img_paths = []
    for idx, path in enumerate(sorted(glob.glob(os.path.join(model_1_results_dir, '*')))):
        model_1_img_paths.append(path)

    model_2_results_dir = args.ttt_dir
    model_2_img_paths = []
    for idx, path in enumerate(sorted(glob.glob(os.path.join(model_2_results_dir, '*')))):
        model_2_img_paths.append(path)

    lr_dir = args.lr_dir
    lr_paths = []
    for idx, path in enumerate(sorted(glob.glob(os.path.join(lr_dir, '*')))):
        lr_paths.append(path)

    test_psnr_vals = defaultdict(list)
    test_ssim_vals = defaultdict(list)
    test_mse_vals = defaultdict(list)
    for i in range(len(test_img_paths)):
        (imgname, imgext) = os.path.splitext(os.path.basename(test_img_paths[i]))
        merged_path = f'{args.merged_dir}/{imgname}_merged{imgext}'
        mr = MergeResults(model_1_img_paths[i], model_2_img_paths[i], test_img_paths[i], lr_paths[i], merged_path, args.scale)
        mr.merge_results()
        for key, val in mr.psnr_values.items():
            if key == 'original':
                test_psnr_vals['original'].append(val[0])
                test_ssim_vals['original'].append(mr.ssim_values[key][0])
                test_mse_vals['original'].append(mr.mse_values[key][0])
                s = f"Original PSNR: {val[0]}; Original SSIM: {mr.ssim_values[key][0]}; Original MSE: {mr.mse_values[key][0]}\n"
                res_strs.append(s)
                print(s)
            elif key == 'ttt':
                test_psnr_vals['ttt'].append(val[0])
                test_ssim_vals['ttt'].append(mr.ssim_values[key][0])
                test_mse_vals['ttt'].append(mr.mse_values[key][0])
                s = f"TTT PSNR: {val[0]}; TTT SSIM: {mr.ssim_values[key][0]}; TTT MSE: {mr.mse_values[key][0]}\n"
                res_strs.append(s)
                print(s)
            else:
                test_psnr_vals['merged'].append(val[0])
                test_ssim_vals['merged'].append(mr.ssim_values[key][0])
                test_mse_vals['merged'].append(mr.mse_values[key][0])
                s = f"Merged PSNR: {val[0]}; Merged SSIM: {mr.ssim_values[key][0]}; Merged MSE: {mr.mse_values[key][0]}\n\n"
                res_strs.append(s)
                print(s)
    s = f'\n\nAverage Statistics for Test Set:\n'
    res_strs.append(s)
    print(s)
    s = f"Original Average PSNR: {np.mean(test_psnr_vals['original'])}; Original Average SSIM: {np.mean(test_ssim_vals['original'])}; Original Average MSE: {np.mean(test_mse_vals['original'])}\n"
    res_strs.append(s)
    print(s)
    s = f"TTT Average PSNR: {np.mean(test_psnr_vals['ttt'])}; TTT Average SSIM: {np.mean(test_ssim_vals['ttt'])}; TTT Average MSE: {np.mean(test_mse_vals['ttt'])}\n"
    res_strs.append(s)
    print(s)
    s = f"Merged Average PSNR: {np.mean(test_psnr_vals['merged'])}; Merged Average SSIM: {np.mean(test_ssim_vals['merged'])}; Merged Average MSE: {np.mean(test_mse_vals['merged'])}\n"
    res_strs.append(s)
    print(s)
    res_strs.append('\n')
    res_file = open(args.results_log, 'w')
    res_file.writelines(res_strs)
    res_file.close()

if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    main(args)
