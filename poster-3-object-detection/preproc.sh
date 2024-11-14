#!/bin/bash
#BSUB -q c02516
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -J preprocessing
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=40GB]"
#BSUB -W 12:00
#BSUB -o preproc%J.out
#BSUB -e preproc%J.err


# Activate the conda environment (replace with the actual environment name)
source ~/venv/project2_venv/bin/activate


# Run the Python script
python /zhome/26/8/209207/02516-intro-to-dl-in-cv/poster-3-object-detection/preprocessing.py
