#!/bin/bash
#BSUB -q gpua100
#BSUB -n 8
#BSUB -gpu "num=1:mode=exclusive_process" 
#BSUB -J training
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=30GB]"
#BSUB -W 05:05
#BSUB -o job_outputs/train%J.out
#BSUB -e job_outputs/train%J.err


EXPERIMENT_1="exp_cls1_reg1"
EXPERIMENT_2="exp_cls0.1_reg0.9"
EPOCHS=100
LEARNING_RATE=1e-4
IOU_THRESHOLD=0.5
CONFIDENCE_THRESHOLD=0.5
WEIGHT_DECAY=1e-5
NUM_IMAGES=5

# Activate the conda environment (replace with the actual environment name)
source ~/venv/project3_venv/bin/activate
#conda activate project-3

python main.py \
    --experiment_name $EXPERIMENT_1 \
    --num_images $NUM_IMAGES \
    --learning_rate $LEARNING_RATE \
    --num_epochs $EPOCHS \
    --iou_threshold $IOU_THRESHOLD \
    --confidence_threshold $CONFIDENCE_THRESHOLD \
    --weight_decay $WEIGHT_DECAY \
    --cls_weight 0.1 \
    --reg_weight 0.9


