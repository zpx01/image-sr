import numpy as np
import os, re
import os.path
import matplotlib.pyplot as plt
import PIL.Image as Image
import sys
sys.path.append('../')
from data_loader import *
from models.network_swinir import SwinIR as net
from IPython.display import display
from utils import utils_calculate_psnr_ssim as util
from main_train_regression import *

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:32'

OUTPUT_DIR='/checkpoints/zeeshan/second_stage/regression/'
TRAIN_HR_DIR='/home/zeeshan/image-sr/trainsets/trainH/DIV2K_train_HR_sub/'
TRAIN_LR_DIR='/home/zeeshan/image-sr/trainsets/trainL/DIV2K_train_LR_bicubic/X4_sub/'
TRAIN_TTT_DIR='/checkpoints/zeeshan/test_time_training/ttt_div2k_trainset/outputs/swinir_classical_sr_x4_train_ttt/'
TRAIN_PRETRAIN_DIR='/checkpoints/zeeshan/test_time_training/ttt_div2k_trainset/outputs/swinir_classical_sr_x4_train_orig/'
TEST_HR_DIR='/home/zeeshan/image-sr/testsets/Set14_kair/original/'
TEST_LR_DIR='/home/zeeshan/image-sr/testsets/Set14_kair/LRbicx4/'
TEST_TTT_DIR='/checkpoints/zeeshan/test_time_training/set14_ttt/outputs/'
TEST_PRETRAIN_DIR='/checkpoints/zeeshan/test_time_training/set14_swinir/outputs/'
OPT_TYPE='Adam'


model_dir = '/checkpoints/zeeshan/second_stage/regression/0.0001_/zeroed'
for idx, path in enumerate(sorted(glob.glob(os.path.join(model_dir, '*.pth')))):
    device = 'cuda:0'
    print('Model Path:', path)
    model = net(upscale=1, in_chans=6, out_chans=3, 
                    img_size=48, window_size=8,
                    img_range=1., depths=[6, 6, 6, 6, 6, 6, 6], 
                    embed_dim=180, num_heads=[6, 6, 6, 6, 6, 6, 6],
                    mlp_ratio=2, upsampler='pixelshuffle', 
                    resi_connection='1conv', rescale_back=False)
        
    model = torch.nn.DataParallel(model, device_ids=[0, 1])
    pretrained_model = torch.load(path)
    model.load_state_dict(pretrained_model)
    model.to(device)
    print("Model successfully loaded!")
    reg_and_psnr(model, 
                TEST_TTT_DIR, 
                TEST_PRETRAIN_DIR, 
                TEST_LR_DIR, 
                TEST_HR_DIR,
                device=device)