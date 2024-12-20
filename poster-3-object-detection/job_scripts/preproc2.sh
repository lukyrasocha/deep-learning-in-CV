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

source /dtu/blackhole/0a/203690/miniconda3/bin/activate

conda activate project-3

# Run the Python script
python preprocessing.py
