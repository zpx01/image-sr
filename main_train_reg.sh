#!/bin/bash

# The SBATCH directives must appear before any executable line in this script.

#SBATCH --qos=low       # QOS (priority).
#SBATCH -N 1               # Number of nodes requested.
#SBATCH -n 1               # Number of tasks (i.e. processes).
#SBATCH --cpus-per-task=8  # Number of cores per task.
#SBATCH --gres=gpu:6       # Number of GPUs.
#SBATCH -t 2-00:00          # Time requested (D-HH:MM).
##SBATCH --nodelist=em1    # Uncomment if you need a specific machine.

# Uncomment this to have Slurm cd to a directory before running the script.
# You can also just run the script from the directory you want to be in.
##SBATCH -D /home/yossi_gandelsman/ttt/image-sr

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

for LR in 0.001 0.0001
do
        python3 main_train_regression.py \
        --lr ${LR} \
        --train_hr_dir ${TRAIN_HR_DIR} \
        --test_hr_dir ${TEST_HR_DIR} \
        --train_pretrain_dir ${TRAIN_PRETRAIN_DIR} \
        --test_pretrain_dir ${TEST_PRETRAIN_DIR} \
        --train_ttt_dir ${TRAIN_TTT_DIR} \
        --test_ttt_dir ${TEST_TTT_DIR} \
        --output_dir ${OUTPUT_DIR}${LR}_${THR} \
        --train_lr_dir ${TRAIN_LR_DIR} \
        --test_lr_dir ${TEST_LR_DIR} \
        --batch_size 40 \
        --epochs 20 \
        --img_size 48 \
        --window_size 8 \
        --opt_type ${OPT_TYPE}
done

# Print completion time.
date

