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

OUTPUT_DIR='/home/yossi_gandelsman/test_time_training/second_stage/'
TRAIN_HR_DIR='/home/yossi_gandelsman/test_time_training/datasets/from_zishan/trainValidH/DIV2K_valid_HR_sub/'
TRAIN_TTT_DIR='/home/yossi_gandelsman/test_time_training/first_stage/ttt/div2k/results/'
TRAIN_PRETRAIN_DIR='/home/yossi_gandelsman/test_time_training/first_stage/pretrained/swinir_classical_sr_x4_1_30_23/'
TEST_HR_DIR='/home/yossi_gandelsman/test_time_training/datasets/from_zishan/Set14_kair/original/'
TEST_TTT_DIR='/home/yossi_gandelsman/test_time_training/first_stage/ttt/set14/results/'
TEST_PRETRAIN_DIR='/home/yossi_gandelsman/test_time_training/first_stage/pretrained/set14_swinir/'

THR=0.01

for LR in 0.001 0.0001 0.00001
do
        CUDA_VISIBLE_DEVICES=6 python3 main_train_classifier.py \
        --threshold ${THR} \
        --lr ${LR} \
        --train_hr_dir ${TRAIN_HR_DIR} \
        --test_hr_dir ${TEST_HR_DIR} \
        --train_pretrain_dir ${TRAIN_PRETRAIN_DIR} \
        --test_pretrain_dir ${TEST_PRETRAIN_DIR} \
        --train_ttt_dir ${TRAIN_TTT_DIR} \
        --test_ttt_dir ${TEST_TTT_DIR} \
        --output_dir ${OUTPUT_DIR}${LR}_${THR}
done

# Print completion time.

