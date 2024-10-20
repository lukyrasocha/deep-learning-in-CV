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

## Setup WanDB 🟡

1. Create an account in [wandb](https://docs.wandb.ai/quickstart/?_gl=1*18kvjf*_ga*MjA0MDY3MTE0NS4xNzI5NDQ4OTMy*_ga_JH1SJHJQXJ*MTcyOTQ0ODkzMS4xLjAuMTcyOTQ0ODkzMi41OS4wLjA.*_ga_GMYDGNGKDT*MTcyOTQ0ODkzMS4xLjAuMTcyOTQ0ODkzMS4wLjAuMA..*_gcl_au*NjI2NjY5MjE4LjE3Mjk0NDg5MzI.)

2. I will invite you to our project

3. Next, get your `WANDB_API_KEY`

4. In HPC run `wandb login`

5. It will ask you for your `WANDB_API_KEY`, paste it there and then you are successfully logged in and you can run `python main.py`

6. Check the runs in Wandb

## Running the code :rocket:

To run the code, run the following:

```bash
python main.py
```

## Project Structure :file_folder:

The project structure:

```
.
├── README.md
├── figures
├── main.py
├── models
│   ├── losses.py
│   ├── models.py
│   └── train.py
├── requirements.txt
├── utils
│   ├── load_data.py
│   ├── logger.py
│   └── visualize.py
```
