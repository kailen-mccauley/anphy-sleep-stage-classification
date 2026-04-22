#!/bin/bash
#SBATCH -JCNN
#SBATCH -N1 --ntasks-per-node=1
#SBATCH --gres=gpu:1 -C V100-32GB
#SBATCH --mem-per-cpu=8G
#SBATCH --cpus-per-task=4
#SBATCH -t20
#SBATCH -oModelReport-%j.out

#SBATCH --mail-type=BEGIN,END,FAIL       # Mail preferences
#SBATCH --mail-user=eyang82@gatech.edu # E-mail address for notifications

module load cuda
nvidia-smi
module load anaconda3
conda activate anphy_sleep_env                   
srun python main.py
