#!/bin/bash

#SBATCH --output="ssformer.log"
#SBATCH --job-name="ssformer"
#SBATCH --gres=gpu:3
#SBATCH --time=4:00:00 # walltime

srun --gres=gpu:3 singularity exec -nv ~/containers/pytorch-24.02 python train_ssformer.py ~/configs/ssformer.cfg --tr_manifest_path ~/datasets/covid19_cough/manifests/unlabeled_train.csv --val_manifest_path ~/datasets/covid19_cough/manifests/unlabeled_val.csv --id _train_200_epochs