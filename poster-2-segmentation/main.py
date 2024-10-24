import torch
import numpy as np

from torchvision import transforms

from utils.load_data import load_data
from utils.logger import logger
from utils.visualize import display_random_images_and_masks, visualize_predictions
from models.train import train_model
from models.models import EncDec, UNet
from models.losses import bce_loss
from utils.transforms import JointTransform
from torch.utils.data import DataLoader


PH2 = True
DRIVE = True
TRAIN = False


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
logger.info(f"Running on {DEVICE}")

# Load data
logger.working_on("Loading data for PH2")

# Adjustable crop size (set to None if you don't want to crop)
CROP_SIZE = (200, 200)  # or None
RESIZE = (256, 256)  # Resize after cropping (if desired)

transform_ph2 = JointTransform(crop_size=CROP_SIZE, resize=RESIZE)
transform_drive = JointTransform(crop_size=CROP_SIZE, resize=RESIZE)

# Load PH2 dataset with augmentation and artificial data (50 synthetic samples)
logger.working_on("Loading data for PH2")
ph2_train_dataset = load_data('ph2', split='train', transform=transform_ph2)
ph2_val_dataset = load_data('ph2', split='val', transform=transform_ph2)
ph2_test_dataset = load_data('ph2', split='test', transform=transform_ph2)

# Data loaders for PH2
ph2_train_loader = DataLoader(ph2_train_dataset, batch_size=16, shuffle=True)
ph2_val_loader = DataLoader(ph2_val_dataset, batch_size=16, shuffle=False)
ph2_test_loader = DataLoader(ph2_test_dataset, batch_size=16, shuffle=False)

# Load DRIVE dataset with augmentation for training
logger.working_on("Loading data for DRIVE")
drive_train_dataset = load_data('drive', split='train', transform=transform_drive)
drive_val_dataset = load_data('drive', split='val', transform=transform_drive)
drive_test_dataset = load_data('drive', split='test', transform=transform_drive)

# Data loaders for DRIVE
drive_train_loader = DataLoader(drive_train_dataset, batch_size=3, shuffle=True)
drive_val_loader = DataLoader(drive_val_dataset, batch_size=3, shuffle=False)
drive_test_loader = DataLoader(drive_test_dataset, batch_size=3, shuffle=False)

logger.success("Data loaded")

# Display some images
display_random_images_and_masks(ph2_train_dataset, figname="ph2_random.png", num_images=3)
display_random_images_and_masks(drive_train_dataset, figname="drive_random.png", num_images=3)
logger.success("Saved example images and masks to 'figures'")

# Simple Encoder-Decoder on PH2

LEARNING_RATE = 0.0001
MAX_EPOCHS = 20 
loss_fn = bce_loss


encdec_ph2_model = EncDec(input_channels=3, output_channels=1)
optimizer = torch.optim.Adam(encdec_ph2_model.parameters(), lr=LEARNING_RATE)

config= {
    "learning_rate": LEARNING_RATE,
    "architecture": "Simple-Encoder-Decoder",
    "dataset": "PH2",
    "epochs": MAX_EPOCHS,
    "loss_fn": "BinaryCrossEntropy",
    "optimizer": "Adam"
}

logger.working_on("Training simple Encoder-Decoder on PH2")
train_model(encdec_ph2_model, ph2_train_loader, ph2_val_loader, loss_fn, optimizer,wandb_config=config, num_epochs=MAX_EPOCHS, device=DEVICE)
visualize_predictions(encdec_ph2_model, ph2_train_loader, DEVICE, figname="ph2_predictions.png", num_images=3)
logger.success("Saved examples of predictions for Enc-Dec of PH2 to 'figures'")

# Simple Encoder-Decoder on DRIVE
LEARNING_RATE = 0.001
MAX_EPOCHS = 20
loss_fn = bce_loss

encdec_drive_model = EncDec(input_channels=3, output_channels=1)
optimizer = torch.optim.Adam(encdec_drive_model.parameters(), lr=LEARNING_RATE)

config= {
    "learning_rate": LEARNING_RATE,
    "architecture": "Simple-Encoder-Decoder",
    "dataset": "DRIVE",
    "epochs": MAX_EPOCHS,
    "loss_fn": "BinaryCrossEntropy",
    "optimizer": "Adam"
}

if TRAIN:
    logger.working_on("Training simple Encoder-Decoder on DRIVE")
    train_model(encdec_drive_model, drive_train_loader, drive_val_loader, loss_fn, optimizer, wandb_config=config, num_epochs= MAX_EPOCHS, device=DEVICE)
    visualize_predictions(encdec_drive_model, drive_train_loader, DEVICE, figname="drive_predictions.png", num_images=3)
    logger.success("Saved examples of predictions for Enc-Dec of DRIVE to 'figures'")
    
    # Simple Encoder-Decoder on UNet

    LEARNING_RATE = 0.001
    MAX_EPOCHS = 20
    loss_fn = bce_loss

    UNetModel = UNet(in_channels=3, num_classes=1)
    optimizer = torch.optim.Adam(UNetModel.parameters(), lr=LEARNING_RATE)

    config= {
        "learning_rate": LEARNING_RATE,
        "architecture": "UNet",
        "dataset": "PH2",
        "epochs": MAX_EPOCHS,
        "loss_fn": "BinaryCrossEntropy",
        "optimizer": "Adam"
    }


    logger.working_on("Training simple UNet on PH2")
    train_model(UNetModel, ph2_train_loader, ph2_val_loader, loss_fn, optimizer,wandb_config=config, num_epochs=MAX_EPOCHS, device=DEVICE)
    visualize_predictions(UNetModel, ph2_train_loader, DEVICE, figname="UNet_predictions.png", num_images=3)
    logger.success("Saved examples of predictions for UNet to 'figures'")

    # Simple UNet on DRIVE
    LEARNING_RATE = 0.001
    MAX_EPOCHS = 20
    loss_fn = bce_loss

    UNetModel = UNet(in_channels=3, num_classes=1)
    optimizer = torch.optim.Adam(UNetModel.parameters(), lr=LEARNING_RATE)

    config= {
        "learning_rate": LEARNING_RATE,
        "architecture": "UNet",
        "dataset": "DRIVE",
        "epochs": MAX_EPOCHS,
        "loss_fn": "BinaryCrossEntropy",
        "optimizer": "Adam"
    }

    logger.working_on("Training simple UNet on DRIVE")
    train_model(UNetModel, drive_train_loader, drive_val_loader, loss_fn, optimizer, wandb_config=config, num_epochs= MAX_EPOCHS, device=DEVICE)
    visualize_predictions(UNetModel, drive_train_loader, DEVICE, figname="drive_predictions_UNet.png", num_images=3)
    logger.success("Saved examples of predictions for UNet of DRIVE to 'figures'")


    # Evaluation

    # Plots
