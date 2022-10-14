import numpy as np
import matplotlib.pyplot as plt 
import matplotlib.image as mpimg
import sys
import matplotlib.colors


def overlay(img1, img2, lr, gt, i):
    b_map = generate_binary_map(img1, img2, gt)
    # print(b_map.shape, img1.shape, img2.shape)
    rows, columns = 3, 2
    fig = plt.figure(figsize=(10, 10))
    cmap = matplotlib.colors.ListedColormap(['red', 'lime'])
    fig.add_subplot(rows, columns, 1)
    plt.imshow(lr)
    plt.title('Original LR')
    fig.add_subplot(rows, columns, 2)
    plt.imshow(gt)
    plt.title('Original HR')
    fig.add_subplot(rows, columns, 3)
    plt.imshow(img1)
    plt.title('2x SwinIR Pretrained')
    fig.add_subplot(rows, columns, 4)
    plt.imshow(img2)
    plt.title('2x SwinIR with TTT')
    fig.add_subplot(rows, columns, 5)
    plt.imshow(gt)
    plt.imshow(b_map, cmap=cmap, alpha=0.45)
    plt.title('Overlayed L2 Loss Bitmap')
    plt.savefig(f"test_time_overlay_x2_{i}.png")

def generate_binary_map(img1, img2, gt):
    binary_map = np.zeros(shape=(img1.shape[0], img1.shape[1]))
    img1_loss, img2_loss = l2_loss(img1, gt), l2_loss(img2, gt)
    for i in range(img1_loss.shape[0]):
        for j in range(img1_loss.shape[1]):
            if img1_loss[i][j] < img2_loss[i][j]:
                binary_map[i][j] = False
            else:
                binary_map[i][j] = True
    return binary_map

def l2_loss(img, gt):
    loss = np.zeros(shape=(img.shape[0], img.shape[1]))
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            pred_val, gt_val = np.array([img[i][j]]), np.array([gt[i][j]])
            l2 = np.sum(np.power((gt_val - pred_val), 2))
            loss[i][j] = l2
    return loss

original_LR_img_paths = [
    '/home/zeeshan/image-sr/testsets/Set5/LRbicx2/baby.png',
    '/home/zeeshan/image-sr/testsets/Set5/LRbicx2/bird.png',
    '/home/zeeshan/image-sr/testsets/Set5/LRbicx2/butterfly.png',
    '/home/zeeshan/image-sr/testsets/Set5/LRbicx2/head.png',
    '/home/zeeshan/image-sr/testsets/Set5/LRbicx2/woman.png',
]

original_HR_img_paths = [
    '/home/zeeshan/image-sr/testsets/Set5/original/baby.png',
    '/home/zeeshan/image-sr/testsets/Set5/original/bird.png',
    '/home/zeeshan/image-sr/testsets/Set5/original/butterfly.png',
    '/home/zeeshan/image-sr/testsets/Set5/original/head.png',
    '/home/zeeshan/image-sr/testsets/Set5/original/woman.png',
]

swinir_img_paths = [
    '/home/zeeshan/image-sr/results/swinir_classical_sr_x2/baby_SwinIR.png',
    '/home/zeeshan/image-sr/results/swinir_classical_sr_x2/bird_SwinIR.png',
    '/home/zeeshan/image-sr/results/swinir_classical_sr_x2/butterfly_SwinIR.png',
    '/home/zeeshan/image-sr/results/swinir_classical_sr_x2/head_SwinIR.png',
    '/home/zeeshan/image-sr/results/swinir_classical_sr_x2/woman_SwinIR.png',
]

swinir_img_paths_ttt = [
    '/home/zeeshan/image-sr/results/swinir_classical_sr_x2/baby_SwinIR_ttt.png',
    '/home/zeeshan/image-sr/results/swinir_classical_sr_x2/bird_SwinIR_ttt.png',
    '/home/zeeshan/image-sr/results/swinir_classical_sr_x2/butterfly_SwinIR_ttt.png',
    '/home/zeeshan/image-sr/results/swinir_classical_sr_x2/head_SwinIR_ttt.png',
    '/home/zeeshan/image-sr/results/swinir_classical_sr_x2/woman_SwinIR_ttt.png'
]

for i in range(len(swinir_img_paths)):
    img1 = mpimg.imread(swinir_img_paths[i])
    img2 = mpimg.imread(swinir_img_paths_ttt[i])
    gt = mpimg.imread(original_HR_img_paths[i])
    lr = mpimg.imread(original_LR_img_paths[i])
    overlay(img1, img2, lr, gt, i)