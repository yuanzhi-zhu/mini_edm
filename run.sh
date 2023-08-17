#!/bin/bash

echo Running on host: `hostname`
echo In directory: `pwd`
echo Starting on: `date`
echo SLURM_JOB_ID: $SLURM_JOB_ID
echo " "

###########################################################################################################

# # Define data set
# DATASET="cifar10"

# # Define model architecture parameters
# CHANNEL_MULT="1 1 1 2"
# MODEL_CHANNELS="128"
# ATTN_RESOLUTIONS="0"
# LAYERS_PER_BLOCK="6"

# Define data set
DATASET="mnist"

# Define model architecture parameters
CHANNEL_MULT="1 2 3 4"
MODEL_CHANNELS="16"
ATTN_RESOLUTIONS="0"
LAYERS_PER_BLOCK="1"

## generation
python -u train_edm.py --dataset $DATASET \
        --channel_mult $CHANNEL_MULT --model_channels $MODEL_CHANNELS \
        --attn_resolutions $ATTN_RESOLUTIONS --layers_per_block $LAYERS_PER_BLOCK \
        --train_batch_size 512 --num_steps 100000 \
        --learning_rate 5e-4 --accumulation_steps 1 \
        --save_images_step 200 \
        --save_model_iters 1000
        

# ## Sampling
# python -u sample_edm.py --dataset $DATASET \
#         --channel_mult $CHANNEL_MULT --model_channels $MODEL_CHANNELS \
#         --attn_resolutions $ATTN_RESOLUTIONS --layers_per_block $LAYERS_PER_BLOCK \
#         --sample_mode save \
#         --eval_batch_size 64 \
#         --model_paths /cluster/home/jinliang/work/ckpts_yuazhu/mini_edm-main/exps/base_20230817-1732


# ## Sampling fid
# python -u sample_edm.py --dataset $DATASET \
#         --channel_mult $CHANNEL_MULT --model_channels $MODEL_CHANNELS \
#         --attn_resolutions $ATTN_RESOLUTIONS --layers_per_block $LAYERS_PER_BLOCK \
#         --sample_mode fid \
#         --num_fid_sample 1000 \
#         --fid_batch_size 256 \
#         --model_paths /scratch_net/rind/yuazhu/exps/base_SongUnet_20230806-0243
        