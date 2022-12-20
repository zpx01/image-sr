import sys, glob, os
import numpy as np
import matplotlib.pyplot as plt 
import matplotlib.image as mpimg
import matplotlib.colors
import argparse

def get_args_parser():
    parser = argparse.ArgumentParser('Overlay Image Visualization', add_help=False)
    parser.add_argument('--thresh', type=float, default=0.001, help='loss threshold for confidence interval')
    parser.add_argument('--pretrained', type=str, help='path to base pretrained model inference images')
    parser.add_argument('--ttt', type=str, help='path to TTT model inference images')
    parser.add_argument('--gt', type=str, help='path to ground truth images')
    parser.add_argument('--lr', type=str, help='path to directory for merged image results')
    parser.add_argument('--results_dir', type=str, help='path to folder to save new images')
    return parser


def overlay(args, img1, img2, lr, gt, i, name):
    b_map = generate_binary_map(args.thresh, img1, img2, gt)
    rows, columns = 3, 2
    fig = plt.figure(figsize=(10, 10))
    cmap = matplotlib.colors.ListedColormap(['red', 'lime', 'beige'])

    fig.add_subplot(rows, columns, 1)
    plt.imshow(lr)
    plt.title('Original LR')
    fig.add_subplot(rows, columns, 2)
    plt.imshow(gt)
    plt.title('Original HR')
    fig.add_subplot(rows, columns, 3)
    plt.imshow(img1)
    plt.title('4x SwinIR Pretrained')
    fig.add_subplot(rows, columns, 4)
    plt.imshow(img2)
    plt.title('4x SwinIR with TTT')
    fig.add_subplot(rows, columns, 5)
    plt.imshow(gt)
    plt.imshow(b_map, cmap=cmap, alpha=0.6)
    plt.title('Overlayed L2 Loss Bitmap (Merged)')
    plt.savefig(f"{args.results_dir}/{name}_overlay.png")

def generate_binary_map(thresh, img1, img2, gt):
    binary_map = np.zeros(shape=(img1.shape[0], img1.shape[1]))
    img1_loss, img2_loss = l2_loss(img1, gt), l2_loss(img2, gt)
    for i in range(img1_loss.shape[0]):
        for j in range(img1_loss.shape[1]):
            if img2_loss[i][j] - img1_loss[i][j] > thresh:
                binary_map[i][j] = 1
            elif img1_loss[i][j] - img2_loss[i][j] > thresh:
                binary_map[i][j] = 0
            else:
                binary_map[i][j] = 2

    return binary_map

def l2_loss(img, gt):
    loss = np.zeros(shape=(img.shape[0], img.shape[1]))
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            pred_val, gt_val = np.array([img[i][j]]), np.array([gt[i][j]])
            l2 = np.sum(np.power((gt_val - pred_val), 2))
            loss[i][j] = l2
    return loss

def main(args):
    gt_folder = args.gt
    gt_paths = []
    for idx, path in enumerate(sorted(glob.glob(os.path.join(gt_folder, '*')))):
        gt_paths.append(path)
    
    ttt_folder = args.ttt
    ttt_paths = []
    for idx, path in enumerate(sorted(glob.glob(os.path.join(ttt_folder, '*')))):
        ttt_paths.append(path)
    
    swinir_folder = args.pretrained
    swinir_paths = []
    for idx, path in enumerate(sorted(glob.glob(os.path.join(swinir_folder, '*')))):
        swinir_paths.append(path)

    lr_folder = args.lr
    lr_paths = []
    for idx, path in enumerate(sorted(glob.glob(os.path.join(lr_folder, '*')))):
        lr_paths.append(path)

    for i in range(len(gt_paths)):
        (imgname, imgext) = os.path.splitext(os.path.basename(gt_paths[i]))
        img1 = mpimg.imread(swinir_paths[i])
        img2 = mpimg.imread(ttt_paths[i])
        gt = mpimg.imread(gt_paths[i])
        lr = mpimg.imread(lr_paths[i])
        overlay(args, img1, img2, lr, gt, i, imgname)

if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    main(args)