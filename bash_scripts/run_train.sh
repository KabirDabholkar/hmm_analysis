#!/usr/bin/bash -l
#eval "$(conda shell.bash hook)"
conda activate nlb-lightning
HYDRA_FULL_ERROR=1 python3 ~/Dropbox/nlb-lightning/scripts_hydra/train.py
#HYDRA_FULL_ERROR=1 python3 ~/Dropbox/nlb-lightning/scripts_hydra/train_ray_tune.py
