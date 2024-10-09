#!/bin/bash

#SBATCH --quotatype=reserved
#SBATCH --job-name=Daformer
#SBATCH --gres=gpu:1
#SBATCH --partition=optimal
#SBATCH -o mean_v9_savepth_%j.out


python -u run_experiments.py --config configs/xxformer/cs2dsec_e2vid_online_semi_xxformer.py
