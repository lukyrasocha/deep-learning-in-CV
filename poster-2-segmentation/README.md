# Project 2: Segmentation :camera:

This project is about image segmentation. The goal is to segment two datasets. The vessel dataset and the PH2 dataset. The vessel dataset is a dataset of retinal images. The PH2 dataset is a dataset of skin lesions. The goal is to segment the vessels in the retinal images and the skin lesions in the PH2 dataset.

## Setup :wrench:

First login to the HPC

**0. SSH setup**

```bash
eval "$(ssh-agent -s)"

ssh-add ~/path_to_your_private_ssh_key
```

After setting up your ssh keys, clone the repository:

**1. Clone the repository**

```bash
git clone git@github.com:lukyrasocha/02516-intro-to-dl-in-cv.git
```

**2. Create a virtual environment**

```bash
python -m venv ~/venv/project2_venv
```

**3. Activate the virtual environment**

```bash
source ~/venv2/project2_venv/bin/activate
```

**4. Change directories**

```bash
cd 02516-intro-to-dl-in-cv/poster-2-segmentation
```
**5. Install the requirements**

```bash
pip install -r requirements.txt
```

## Running the code :rocket:

To run the code, run the following:

```bash
python main.py
```

## Project Structure :file_folder:

The project structure:

```
.
├── utils
│   ├── load_data.py
│   ├── metrics.py
│   ├── logger.py

├── models
│   ├── unet.py
│   ├── encoder_decoder.py

├── main.py
```
