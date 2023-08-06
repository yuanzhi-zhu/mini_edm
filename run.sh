#!/bin/bash

source /scratch_net/rind/yuazhu/pt1100/bin/activate

echo Running on host: `hostname`
echo In directory: `pwd`
echo Starting on: `date`
echo SLURM_JOB_ID: $SLURM_JOB_ID
echo " "

###########################################################################################################

## generation
python -u edm.py --train_batch_size 64 --num_steps 200000 \
        --learning_rate 1e-4 --accumulation_steps 4 \
        --save_images_step 500 --architecture SongUnet \
        --save_model_iters 2500
        
