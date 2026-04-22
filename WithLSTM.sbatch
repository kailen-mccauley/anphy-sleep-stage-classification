#!/bin/bash
#SBATCH -JWithLSTM
#SBATCH -N1 --ntasks-per-node=1
#SBATCH -G1 
#SBATCH --mem-per-cpu=2G
#SBATCH --cpus-per-task=8
#SBATCH -t360
#SBATCH -oModelReportWithLSTM-%j.out

#SBATCH --mail-type=BEGIN,END,FAIL       # Mail preferences
#SBATCH --mail-user=eyang82@gatech.edu # E-mail address for notifications

module load cuda
nvidia-smi
module load anaconda3
conda activate anphy_sleep_env                   
srun python main.py
