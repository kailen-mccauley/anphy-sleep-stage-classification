#!/bin/bash
#SBATCH -JLSTM_ADAM_only_body
#SBATCH -N1 --ntasks-per-node=1
#SBATCH --gres=gpu:1 
#SBATCH --constraint="L40S|HX00|A40|A100-40GB|A100-80GB|V100-16GB|V100-32GB|RTX6000"
#SBATCH --mem-per-cpu=10G
#SBATCH --cpus-per-task=8
#SBATCH -t210
#SBATCH -oCNNWithBiLSTM_ADAM_only_body-%j.out

#SBATCH --mail-type=BEGIN,END,FAIL       # Mail preferences
#SBATCH --mail-user=eyang82@gatech.edu # E-mail address for notifications

module load cuda
nvidia-smi
module load anaconda3
conda activate anphy_sleep_env                   
srun python main.py --config ./configs/LSTM_ADAM_only_body.yaml
