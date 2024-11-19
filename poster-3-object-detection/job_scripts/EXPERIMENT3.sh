#!/bin/bash
#BSUB -q c02516
#BSUB -n 8
#BSUB -gpu "num=1:mode=exclusive_process" 
#BSUB -J REG_10_CLF_1
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=20GB]"
#BSUB -W 12:00
#BSUB -o job_outputs/REG_10_CLF_1%J.out
#BSUB -e job_outputs/REG_10_CLF_1%J.err

EXPERIMENT="EXPERIMENT3Reg10"
EPOCHS=50
LEARNING_RATE=1e-4
IOU_THRESHOLD=0.3
CONFIDENCE_THRESHOLD=0.5
WEIGHT_DECAY=1e-5
NUM_IMAGES=10

source ~/venv/project3_venv/bin/activate
#conda activate project-3

python main.py \
    --experiment_name $EXPERIMENT \
    --num_images $NUM_IMAGES \
    --learning_rate $LEARNING_RATE \
    --num_epochs $EPOCHS \
    --iou_threshold $IOU_THRESHOLD \
    --confidence_threshold $CONFIDENCE_THRESHOLD \
    --weight_decay $WEIGHT_DECAY \
    --cls_weight 1.0 \
    --reg_weight 10.0