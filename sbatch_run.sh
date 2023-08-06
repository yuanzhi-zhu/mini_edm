#!/bin/bash

conda deactivate
cd /scratch_net/rind/yuazhu

# sbatch -n 8 --mem-per-cpu=10G --gpus=1 --gres=gpumem:12g --constraint='titan_xp|geforce_gtx_titan_x' \
#         --time=24:00:00 < /scratch_net/rind/yuazhu/run.sh

sbatch --gres=gpu:1 --time=24:00:00 --output=/scratch_net/rind/yuazhu/log/%j.out --mem=30G /scratch_net/rind/yuazhu/run.sh --constraint='rtx_2080_ti|titan_xp|gtx_titan_x|geforce_gtx_titan_x'

# lmod2env && srun --time=24:00:00 -n 6 --mem-per-cpu=5G -G 1 --gres=gpumem:20G  --constraint='titan_xp|geforce_gtx_titan_x' --pty bash -i