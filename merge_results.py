import os, glob, argparse
import numpy as np
from data_loader_utils import *
import matplotlib.image as mpimg
from collections import defaultdict
import torch
import skimage

res_strs = []
class MergeResults(object):
    """
    MergeResults takes outputs of two classical SR models, one with test-time
    training adaptation and one without. It merges the outputs of the two models
    by taking the best pixels (determined by L2 loss) from both images to form
    a new, merged inference. It provides the PSNR and SSIM for each of the three
    image categories.
    """
    def __init__(self, model_path_1, model_path_2, gt_path, merged_img_path):
        self.model_path_1 = model_path_1 # original pretrained model inference
        self.model_path_2 = model_path_2 # ttt model inference 
        self.gt_path = gt_path
        self.merged_img_path = merged_img_path
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.psnr_values = defaultdict(list)
        self.ssim_values = defaultdict(list)
    
    def merge_results(self):
        # load images
        global res_strs
        img1 = mpimg.imread(self.model_path_1)
        img2 = mpimg.imread(self.model_path_2)
        gt = mpimg.imread(self.gt_path)

        (imgname, imgext) = os.path.splitext(os.path.basename(self.gt_path))
        s = f"Testing {imgname} ({self.gt_path})...\n"
        print(s)
        res_strs.append(s)

        # compute individual PSNR/SSIM values
        original_psnr, ttt_psnr = self.psnr(gt, img1), self.psnr(gt, img2)
        original_ssim, ttt_ssim = self.ssim(gt, img1), self.ssim(gt, img2)

        # create merged image
        merged_img = np.zeros(shape=gt.shape, dtype=np.float32)
        for i in range(len(img1)):
            for j in range(len(img1[i])):
                v1, v2, gt_v = img1[i][j], img2[i][j], gt[i][j]
                norm1, norm2 = self.l2_norm_pixel(gt_v, v1), self.l2_norm_pixel(gt_v, v2)
                if norm1 <= norm2:
                    merged_img[i][j] = img1[i][j]
                elif norm2 < norm1:
                    merged_img[i][j] = img2[i][j]

        # save merged image
        mpimg.imsave(self.merged_img_path, merged_img)
        
        # compute merged PSNR/SSIM values
        merged_psnr = self.psnr(gt, merged_img)
        merged_ssim = self.ssim(gt, merged_img)

        # save metrics
        self.psnr_values["original"].append(original_psnr)
        self.psnr_values["ttt"].append(ttt_psnr)
        self.psnr_values["merged"].append(merged_psnr)
        self.ssim_values["original"].append(original_ssim)
        self.ssim_values["ttt"].append(ttt_ssim)
        self.ssim_values["merged"].append(merged_ssim)

    def psnr(self, img_gt, output):
        psnr = skimage.metrics.peak_signal_noise_ratio(img_gt, output)
        psnr = float('{:.2f}'.format(psnr))
        return psnr
    
    def ssim(self, img_gt, output):
        ssim = skimage.metrics.structural_similarity(img_gt, output, channel_axis=1, multichannel=True)
        ssim = float('{:.2f}'.format(ssim))
        return ssim

    def l2_norm_pixel(self, v1, v2):
        return np.sum((v1 - v2)**2)


def get_args_parser():
    parser = argparse.ArgumentParser('MergeResults - Classical SR', add_help=False)
    parser.add_argument('--pretrained_dir', type=str, help='path to base pretrained model inference images')
    parser.add_argument('--ttt_dir', type=str, help='path to TTT model inference images')
    parser.add_argument('--gt_dir', type=str, help='path to ground truth images')
    parser.add_argument('--merged_dir', type=str, help='path to directory for merged image results')
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

    test_psnr_vals = defaultdict(list)
    test_ssim_vals = defaultdict(list)
    for i in range(len(test_img_paths)):
        (imgname, imgext) = os.path.splitext(os.path.basename(test_img_paths[i]))
        merged_path = f'{args.merged_dir}/{imgname}_merged{imgext}'
        mr = MergeResults(model_1_img_paths[i], model_2_img_paths[i], test_img_paths[i], merged_path)
        mr.merge_results()
        for key, val in mr.psnr_values.items():
            if key == 'original':
                test_psnr_vals['original'].append(val[0])
                test_ssim_vals['original'].append(mr.ssim_values[key][0])
                s = f"Original PSNR: {val[0]}; Original SSIM: {mr.ssim_values[key][0]}\n"
                res_strs.append(s)
                print(s)
            elif key == 'ttt':
                test_psnr_vals['ttt'].append(val[0])
                test_ssim_vals['ttt'].append(mr.ssim_values[key][0])
                s = f"TTT PSNR: {val[0]}; TTT SSIM: {mr.ssim_values[key][0]}\n"
                res_strs.append(s)
                print(s)
            else:
                test_psnr_vals['merged'].append(val[0])
                test_ssim_vals['merged'].append(mr.ssim_values[key][0])
                s = f"Merged PSNR: {val[0]}; Merged SSIM: {mr.ssim_values[key][0]}\n\n"
                res_strs.append(s)
                print(s)
    s = f'\n\nAverage Statistics for Test Set:\n'
    res_strs.append(s)
    print(s)
    s = f"Original Average PSNR: {np.mean(test_psnr_vals['original'])}; Original Average SSIM: {np.mean(test_ssim_vals['original'])}\n"
    res_strs.append(s)
    print(s)
    s = f"TTT Average PSNR: {np.mean(test_psnr_vals['ttt'])}; TTT Average SSIM: {np.mean(test_ssim_vals['ttt'])}\n"
    res_strs.append(s)
    print(s)
    s = f"Merged Average PSNR: {np.mean(test_psnr_vals['merged'])}; Merged Average SSIM: {np.mean(test_ssim_vals['merged'])}\n"
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
