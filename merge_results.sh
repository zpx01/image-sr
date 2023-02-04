#!/bin/bash

# The SBATCH directives must appear before any executable line in this script.

#SBATCH --qos low       # QOS (priority).
#SBATCH -N 1               # Number of nodes requested.
#SBATCH -n 1               # Number of tasks (i.e. processes).
#SBATCH --cpus-per-task=8  # Number of cores per task.
#SBATCH --gres=gpu:1       # Number of GPUs.
#SBATCH -t 2-12:00          # Time requested (D-HH:MM).
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

echo "Starting SwinIR/TTT Merge job..."

source ~/.bashrc
conda activate image-sr

# Python will buffer output of your script unless you set this.
# If you're not using python, figure out how to turn off output
# buffering when stdout is a file, or else when watching your output
# script you'll only get updated every several lines printed.
export PYTHONUNBUFFERED=1

# Merge SwinIR and TTT inference results

PRETRAINED_DIR='/checkpoints/zeeshan/test_time_training/set14_swinir/swinir_classical_sr_x4_1_30_23' # Directory with pretrained inference images
TTT_DIR='/checkpoints/zeeshan/test_time_training/set14_ttt/swinir_classical_sr_x4_1_30_23_ttt' # Directory with TTT inference images
GT_DIR='/home/zeeshan/image-sr/testsets/Set14_kair/original' # Directory with ground truth images
MERGED_DIR='/checkpoints/zeeshan/test_time_training/set14_merged' # Directory to store merged results in
LR_DIR='/home/zeeshan/image-sr/testsets/Set14_kair/LRbicx4' # Directory with low-res images
RESULTS_LOG='/checkpoints/zeeshan/test_time_training/set14_merged/merged_results.txt' # File path for text file to store result metrics
SCALE=4 # Super-Res Scale Factor
python3 merge_results.py \
        --pretrained_dir ${PRETRAINED_DIR} \
        --ttt_dir ${TTT_DIR} \
        --gt_dir ${GT_DIR} \
        --merged_dir ${MERGED_DIR} \
	--lr_dir ${LR_DIR} \
	--scale ${SCALE} \
        --results_log ${RESULTS_LOG}

date


