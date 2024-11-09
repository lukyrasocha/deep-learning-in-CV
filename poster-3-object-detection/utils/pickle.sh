#!/bin/sh
### ------------ specify queue name ----------------
#BSUB -q c02516

### ------------ specify GPU request----------------
#BSUB -gpu "num=1:mode=exclusive_process"

### ------------ specify job name ------------------
#BSUB -J drive_encdec_no_padding_bce

### ------------ specify number of cores -----------
#BSUB -n 4
#BSUB -R "span[hosts=1]"

### ------------ specify CPU memory requirements ---
#BSUB -R "rusage[mem=20GB]"

### ------------ specify wall-clock time (max allowed is 12:00) ------
#BSUB -W 07:00

### ------------ specify output and error files ----
#BSUB -o OUTPUT_FILE%J.out
#BSUB -e OUTPUT_FILE%J.err

# Load the environment
source ~/venv/project2_venv/bin/activate

# Execute the Python script
python ~/02516-intro-to-dl-in-cv/poster-3-object-detection/utils/load_data_new.py