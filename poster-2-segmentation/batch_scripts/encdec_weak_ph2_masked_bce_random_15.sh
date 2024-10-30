#!/bin/sh  
#BSUB -q c02516
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -J ph2_masked_bce_weaph2_masked_bce_weak_15
#BSUB -n 4  
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=20GB]"
#BSUB -W 12:00  # Max time, adjust if needed
#BSUB -o ph2_masked_bce_weaph2_masked_bce_weak_15%J.out  
#BSUB -e ph2_masked_bce_weaph2_masked_bce_weak_15%J.err

# Load the environment  
source ~/venv/project2_venv/bin/activate

# Execute the Python script with specified parameters  
python ~/02516-intro-to-dl-in-cv/poster-2-segmentation/main.py --model encdec --data ph2 --loss_fn masked_bce --epochs 1500 --padding 1 --visualize --weak --jobid $LSB_JOBID --num_clicks 15  --sampling_strategy random
