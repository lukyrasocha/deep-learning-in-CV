#!/bin/bash
#BSUB -q gpua100
#BSUB -n 8
#BSUB -gpu "num=2:mode=exclusive_process" 
#BSUB -J training
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=10GB]"
#BSUB -W 00:05
#BSUB -o job_outputs/train%J.out
#BSUB -e job_outputs/train%J.err


# c02516
# Activate the conda environment (replace with the actual environment name)
source /dtu/blackhole/0a/203690/miniconda3/bin/activate
conda activate project-3


# Run the Python script
python main.py
