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
TRAIN_HR_DIR='/home/yossi_gandelsman/test_time_training/first_stage/DIV2K_valid_HR_sub/'
TRAIN_TTT_DIR='/home/yossi_gandelsman/test_time_training/first_stage/swinir_classical_sr_x4_1_24_23_ttt/'
TRAIN_PRETRAIN_DIR='/home/yossi_gandelsman/test_time_training/first_stage/swinir_classical_sr_x4_1_24_23_orig/'
TEST_HR_DIR='/home/yossi_gandelsman/test_time_training/first_stage/DIV2K_valid_HR_sub/'
TEST_TTT_DIR='/home/yossi_gandelsman/test_time_training/first_stage/swinir_classical_sr_x4_1_24_23_ttt/'
TEST_PRETRAIN_DIR='/home/yossi_gandelsman/test_time_training/first_stage/swinir_classical_sr_x4_1_24_23_orig/'

python3 main_train_classifier.py \
        --threshold 0.01 \
        --train_hr_dir ${TRAIN_HR_DIR} \
        --test_hr_dir ${TEST_HR_DIR} \
        --train_pretrain_dir ${TRAIN_PRETRAIN_DIR} \
        --test_pretrain_dir ${TEST_PRETRAIN_DIR} \
        --train_ttt_dir ${TRAIN_TTT_DIR} \
        --test_ttt_dir ${TEST_TTT_DIR} \
        --output_dir ${OUTPUT_DIR}
# Print completion time.

