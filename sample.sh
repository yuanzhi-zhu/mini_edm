#!/bin/bash

source /scratch_net/rind/yuazhu/pt1100/bin/activate

echo Running on host: `hostname`
echo In directory: `pwd`
echo Starting on: `date`
echo SLURM_JOB_ID: $SLURM_JOB_ID
echo " "

###########################################################################################################

## generation
python -u sample_edm.py \
        --num_fid_sample 1000 \
        --architecture SongUnet \
        --fid_batch_size 256 \
        --sample_mode save \
        --model_paths /scratch_net/rind/yuazhu/exps/base_SongUnet_20230806-0243
        
        