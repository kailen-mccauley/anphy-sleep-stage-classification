#!/bin/bash
#SBATCH -JDebuggingCUDA
#SBATCH -N1 --ntasks-per-node=1
#SBATCH --gres=gpu:V100:1
#SBATCH --mem-per-gpu=32GB
#SBATCH --cpus-per-task=8
#SBATCH -t20
#SBATCH -oDebuggingCUDA-%j.out

#SBATCH --mail-type=BEGIN,END,FAIL       # Mail preferences
#SBATCH --mail-user=eyang82@gatech.edu # E-mail address for notifications

module load cuda
nvidia-smi
module load anaconda3
conda activate anphy_sleep_env                   
srun python debugging_cuda.py