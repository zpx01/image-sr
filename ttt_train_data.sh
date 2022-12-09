#!/bin/bash

# The SBATCH directives must appear before any executable line in this script.

# SBATCH --qos high6       # QOS (priority).
#SBATCH -N 1               # Number of nodes requested.
#SBATCH -n 1               # Number of tasks (i.e. processes).
#SBATCH --cpus-per-task=8  # Number of cores per task.
#SBATCH --gres=gpu:4       # Number of GPUs.
#SBATCH -t 3-12:00          # Time requested (D-HH:MM).
##SBATCH --nodelist=em7    # Uncomment if you need a specific machine.

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

# Do all the research.
MODEL_PATH='/checkpoints/yossi_gandelsman/image-sr/swinir_sr_classical_patch48_x4/655000_G.pth'
OPTIMIZER_PATH='/checkpoints/yossi_gandelsman/image-sr/swinir_sr_classical_patch48_x4/655000_optimizerG.pth'
TESTSET_DIR='/old_home_that_will_be_deleted_at_some_point/yossi_gandelsman/datasets/SR/trainsets/trainL/DIV2K_train_LR_bicubic/X4_sub/'
OUTPUT_DIR='/checkpoints/yossi_gandelsman/image-sr/swinir_sr_ttt_train_patch48_x4_train/'
python3 main_test_time.py \
        --model_path ${MODEL_PATH} \
        --opt_path ${OPTIMIZER_PATH} \
        --scale 4 \
        --num_images 10 \
        --epochs 5 \
        --test_dir ${TESTSET_DIR} \
        --output_dir ${OUTPUT_DIR}
# Print completion time.


TASK='classical_sr'
TYPE='ttt'
MODELS_DIR='/checkpoints/yossi_gandelsman/image-sr/swinir_sr_ttt_train_patch48_x4/' # Directory containing all TTT checkpoints to test
TEST_FOLDER_LQ='/old_home_that_will_be_deleted_at_some_point/yossi_gandelsman/datasets/SR/trainsets/trainL/DIV2K_train_LR_bicubic/X4_sub' # Low quality images for testing
TEST_FOLDER_GT='/old_home_that_will_be_deleted_at_some_point/yossi_gandelsman/datasets/SR/trainsets/trainH/DIV2K_train_HR' # High quality ground truth images
RESULTS_PATH='/old_home_that_will_be_deleted_at_some_point/yossi_gandelsman/datasets/SR/results/train/' # Path to text file to save metrics
IMG_ID='12_6_22_ttt' # Unique identifier to use for saved image file paths
python3 main_test_swinir.py \
        --task ${TASK} \
        --type ${TYPE} \
        --scale 4 \
        --training_patch_size 48 \
        --models_dir ${MODELS_DIR} \
        --folder_lq ${TEST_FOLDER_LQ} \
        --folder_gt ${TEST_FOLDER_GT} \
        --results_path ${RESULTS_PATH} \
        --img_identifier ${IMG_ID}

TASK='classical_sr'
TYPE='swinir'
MODEL_PATH='/checkpoints/yossi_gandelsman/image-sr/swinir_sr_classical_patch48_x4/655000_G.pth' # SwinIR pretrained model path
TEST_FOLDER_LQ='/old_home_that_will_be_deleted_at_some_point/yossi_gandelsman/datasets/SR/trainsets/trainL/DIV2K_train_LR_bicubic/X4' # Low quality images for testing
TEST_FOLDER_GT='/old_home_that_will_be_deleted_at_some_point/yossi_gandelsman/datasets/SR/trainsets/trainH/DIV2K_train_HR' # High quality ground truth images
RESULTS_PATH='/old_home_that_will_be_deleted_at_some_point/yossi_gandelsman/datasets/SR/results/train/' # Path to text file to save metrics
IMG_ID='12_6_22_orig' # Unique identifier to use for saved image file paths
python3 main_test_swinir.py \
        --task ${TASK} \
        --type ${TYPE} \
        --scale 4 \
        --training_patch_size 48 \
        --model_path ${MODEL_PATH} \
        --folder_lq ${TEST_FOLDER_LQ} \
        --folder_gt ${TEST_FOLDER_GT} \
        --results_path ${RESULTS_PATH} \
        --img_identifier ${IMG_ID}