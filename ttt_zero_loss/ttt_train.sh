#!/bin/bash

# The SBATCH directives must appear before any executable line in this script.

#SBATCH --qos low       # QOS (priority).
#SBATCH -N 1               # Number of nodes requested.
#SBATCH -n 1               # Number of tasks (i.e. processes).
#SBATCH --cpus-per-task=8  # Number of cores per task.
#SBATCH --gres=gpu:6       # Number of GPUs.
#SBATCH -t 2-00:00          # Time requested (D-HH:MM).
##SBATCH --nodelist=em3    # Uncomment if you need a specific machine.

# Uncomment this to have Slurm cd to a directory before running the script.
# You can also just run the script from the directory you want to be in.
##SBATCH -D /home/USERID/path/to/something

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

MODEL_PATH='/home/zeeshan/image-sr/superresolution/swinir_sr_classical_patch48_x4/models/655000_G.pth'
OPTIMIZER_PATH='/home/zeeshan/image-sr/superresolution/swinir_sr_classical_patch48_x4/models/655000_optimizerG.pth'
TESTSET_DIR='/home/zeeshan/image-sr/testsets/Set5/LR_bicubic/X4'
BATCH_SIZE=2
ZERO_LOSS=TRUE
SAVE_FREQ=20
OUTPUT_DIR='/checkpoints/zeeshan/test_time_training/zero_loss'
python3 /home/zeeshan/image-sr/main_test_time.py \
        --model_path ${MODEL_PATH} \
        --opt_path ${OPTIMIZER_PATH} \
        --scale 4 \
        --num_images 15 \
        --epochs 5 \
        --test_dir ${TESTSET_DIR} \
        --output_dir ${OUTPUT_DIR} \
	--batch_size ${BATCH_SIZE} \
	--zero_loss ${ZERO_LOSS} \
        --save_freq ${SAVE_FREQ}

date