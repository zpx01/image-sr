#!/bin/bash

# The SBATCH directives must appear before any executable line in this script.

# SBATCH --qos high6       # QOS (priority).
#SBATCH -N 1               # Number of nodes requested.
#SBATCH -n 1              # Number of tasks (i.e. processes).
#SBATCH --cpus-per-task=8  # Number of cores per task.
#SBATCH --gres=gpu:1       # Number of GPUs.
#SBATCH -t 3-12:00          # Time requested (D-HH:MM).
##SBATCH --nodelist=em6    # Uncomment if you need a specific machine.

# Uncomment this to have Slurm cd to a directory before running the script.
# You can also just run the script from the directory you want to be in.
#SBATCH -D /home/yossi_gandelsman/ttt/image-sr

# Uncomment to control the output files. By default stdout and stderr go to
# the same place, but if you use both commands below they'll be split up.
# %N is the hostname (if used, will create output(s) per node).
# %j is jobid.
##SBATCH -o slurm.%N.%j.out    # STDOUT
##SBATCH -e slurm.%N.%j.err    # STDERR

# Print some info for context.
pwd
hostname
date
nvidia-smi

echo "Starting SwinIR training job (classical SR)..."

source ~/.bashrc
conda activate image-sr

# Python will buffer output of your script unless you set this.
# If you're not using python, figure out how to turn off output
# buffering when stdout is a file, or else when watching your output
# script you'll only get updated every several lines printed.
export PYTHONUNBUFFERED=1


# Train the TTT for all the images in DIV2K_train_LR_bicubic/X4_sub
MODEL_PATH='/home/zeeshan/image-sr/superresolution/swinir_sr_classical_patch48_x4/models/classicalSR_SwinIR_x4.pth'
OPTIMIZER_PATH='/'
TESTSET_DIR='/home/zeeshan/image-sr/testsets/Set14_kair/LRbicx4'
BATCH_SIZE=4
ZERO_LOSS=TRUE
SAVE_FREQ=10000
NO_OPT=TRUE
OUTPUT_DIR='/checkpoints/zeeshan/test_time_training/set14_ttt/sgd_models'

python3 main_test_time.py \
        --model_path ${MODEL_PATH} \
        --opt_path ${OPTIMIZER_PATH} \
        --scale 4 \
        --num_images 15 \
        --epochs 5 \
        --test_dir ${TESTSET_DIR} \
        --output_dir ${OUTPUT_DIR} \
        --batch_size ${BATCH_SIZE} \
        --zero_loss ${ZERO_LOSS} \
        --save_freq ${SAVE_FREQ} \
        --no_opt ${NO_OPT} \
        --device 'cuda:0' 
        # &
# python3 main_test_time.py \
#         --model_path ${MODEL_PATH} \
#         --opt_path ${OPTIMIZER_PATH} \
#         --scale 4 \
#         --num_images 15 \
#         --epochs 5 \
#         --test_dir ${TESTSET_DIR} \
#         --output_dir ${OUTPUT_DIR} \
#         --batch_size ${BATCH_SIZE} \
#         --zero_loss ${ZERO_LOSS} \
#         --save_freq ${SAVE_FREQ} \
#         --no_opt ${NO_OPT} \
#         --device 'cuda:1' &
# python3 main_test_time.py \
#         --model_path ${MODEL_PATH} \
#         --opt_path ${OPTIMIZER_PATH} \
#         --scale 4 \
#         --num_images 15 \
#         --epochs 5 \
#         --test_dir ${TESTSET_DIR} \
#         --output_dir ${OUTPUT_DIR} \
#         --batch_size ${BATCH_SIZE} \
#         --zero_loss ${ZERO_LOSS} \
#         --save_freq ${SAVE_FREQ} \
#         --no_opt ${NO_OPT} \
#         --device 'cuda:2' &
# python3 main_test_time.py \
#         --model_path ${MODEL_PATH} \
#         --opt_path ${OPTIMIZER_PATH} \
#         --scale 4 \
#         --num_images 15 \
#         --epochs 5 \
#         --test_dir ${TESTSET_DIR} \
#         --output_dir ${OUTPUT_DIR} \
#         --batch_size ${BATCH_SIZE} \
#         --zero_loss ${ZERO_LOSS} \
#         --save_freq ${SAVE_FREQ} \
#         --no_opt ${NO_OPT} \
#         --device 'cuda:3' &
# python3 main_test_time.py \
#         --model_path ${MODEL_PATH} \
#         --opt_path ${OPTIMIZER_PATH} \
#         --scale 4 \
#         --num_images 15 \
#         --epochs 5 \
#         --test_dir ${TESTSET_DIR} \
#         --output_dir ${OUTPUT_DIR} \
#         --batch_size ${BATCH_SIZE} \
#         --zero_loss ${ZERO_LOSS} \
#         --save_freq ${SAVE_FREQ} \
#         --no_opt ${NO_OPT} \
#         --device 'cuda:4' &
# python3 main_test_time.py \
#         --model_path ${MODEL_PATH} \
#         --opt_path ${OPTIMIZER_PATH} \
#         --scale 4 \
#         --num_images 15 \
#         --epochs 5 \
#         --test_dir ${TESTSET_DIR} \
#         --output_dir ${OUTPUT_DIR} \
#         --batch_size ${BATCH_SIZE} \
#         --zero_loss ${ZERO_LOSS} \
#         --save_freq ${SAVE_FREQ} \
#         --no_opt ${NO_OPT} \
#         --device 'cuda:5'
date


# Remove leftovers

# Evaluate the ttt model
# TASK='classical_sr'
# TYPE='ttt'
# MODELS_DIR='/checkpoints/yossi_gandelsman/image-sr/swinir_sr_ttt_train_patch48_div2k_lr_bicubic_x4_sub/' # Directory containing all TTT checkpoints to test
# TEST_FOLDER_LQ='/old_home_that_will_be_deleted_at_some_point/yossi_gandelsman/datasets/SR/trainsets/trainL/DIV2K_train_LR_bicubic/X4_sub' # Low quality images for testing
# TEST_FOLDER_GT='/old_home_that_will_be_deleted_at_some_point/yossi_gandelsman/datasets/SR/trainsets/trainH/DIV2K_train_HR' # High quality ground truth images
# RESULTS_PATH='/old_home_that_will_be_deleted_at_some_point/yossi_gandelsman/datasets/SR/results/train/' # Path to text file to save metrics
# IMG_ID='12_21_22_ttt' # Unique identifier to use for saved image file paths
# python3 main_test_swinir.py \
#         --task ${TASK} \
#         --type ${TYPE} \
#         --scale 4 \
#         --training_patch_size 48 \
#         --models_dir ${MODELS_DIR} \
#         --folder_lq ${TEST_FOLDER_LQ} \
#         --folder_gt ${TEST_FOLDER_GT} \
#         --results_path ${RESULTS_PATH} \
#         --img_identifier ${IMG_ID}

# Evaluate the original model
# TASK='classical_sr'
# TYPE='swinir'
# MODEL_PATH='/checkpoints/yossi_gandelsman/image-sr/swinir_sr_classical_patch48_x4/655000_G.pth' # SwinIR pretrained model path
# TEST_FOLDER_LQ='/old_home_that_will_be_deleted_at_some_point/yossi_gandelsman/datasets/SR/trainsets/trainL/DIV2K_train_LR_bicubic/X4' # Low quality images for testing
# TEST_FOLDER_GT='/old_home_that_will_be_deleted_at_some_point/yossi_gandelsman/datasets/SR/trainsets/trainH/DIV2K_train_HR' # High quality ground truth images
# RESULTS_PATH='/old_home_that_will_be_deleted_at_some_point/yossi_gandelsman/datasets/SR/results/train/' # Path to text file to save metrics
# IMG_ID='12_21_22_orig' # Unique identifier to use for saved image file paths
# python3 main_test_swinir.py \
#         --task ${TASK} \
#         --type ${TYPE} \
#         --scale 4 \
#         --training_patch_size 48 \
#         --model_path ${MODEL_PATH} \
#         --folder_lq ${TEST_FOLDER_LQ} \
#         --folder_gt ${TEST_FOLDER_GT} \
#         --results_path ${RESULTS_PATH} \
#         --img_identifier ${IMG_ID}