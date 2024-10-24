import torch
import numpy as np

from torchvision import transforms

from utils.load_data import load_data
from utils.logger import logger
from utils.visualize import display_random_images_and_masks, visualize_predictions
from models.train import train_model
from models.models import EncDec
from models.losses import bce_loss, focal_loss
from models.models import DoubleConv, DownSample, UpSample, UNet




DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
logger.info(f"Running on {DEVICE}")

# Load data
logger.working_on("Loading data for PH2")

transform_ph2 = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

transform_drive = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

ph2_train_dataset = load_data('ph2', split='train', transform=transform_ph2)
ph2_val_dataset = load_data('ph2', split='val', transform=transform_ph2)
ph2_test_dataset = load_data('ph2', split='test', transform=transform_ph2)

print('PH2 Train Dataset size:', len(ph2_train_dataset))
print('PH2 Validation Dataset size:', len(ph2_val_dataset))
print('PH2 Test Dataset size:', len(ph2_test_dataset))

image, mask = ph2_train_dataset[1]
print('Image shape:', image.shape)
print('Mask shape:', mask.shape)
assert len(np.unique(mask.numpy()[0])) == 2, "mask needs to have binary values (0,1)"

# Data loaders
ph2_train_loader = torch.utils.data.DataLoader(ph2_train_dataset, batch_size=16, shuffle=True)
ph2_val_loader = torch.utils.data.DataLoader(ph2_val_dataset, batch_size=16, shuffle=False)
ph2_test_loader = torch.utils.data.DataLoader(ph2_test_dataset, batch_size=16, shuffle=False)

logger.working_on("Loading data for DRIVE")

drive_train_dataset = load_data('drive', split='train', transform=transform_drive)
drive_val_dataset = load_data('drive', split='val', transform=transform_drive)
drive_test_dataset = load_data('drive', split='test', transform=transform_drive)

print('DRIVE Train Dataset size:', len(drive_train_dataset))
print('DRIVE Validation Dataset size:', len(drive_val_dataset))
print('DRIVE Test Dataset size:', len(drive_test_dataset))

# Data loaders
drive_train_loader = torch.utils.data.DataLoader(drive_train_dataset, batch_size=3, shuffle=True)
drive_val_loader = torch.utils.data.DataLoader(drive_val_dataset, batch_size=3, shuffle=False)
drive_test_loader = torch.utils.data.DataLoader(drive_test_dataset, batch_size=3, shuffle=False)

image, mask = drive_train_dataset[1]
print('Image shape:', image.shape)
print('Mask shape:', mask.shape)
assert len(np.unique(mask.numpy()[0])) == 2, "mask needs to have binary values (0,1)"

logger.success("Data loaded")

# Display some images
display_random_images_and_masks(ph2_train_dataset, figname="ph2_random.png", num_images=3)
display_random_images_and_masks(drive_train_dataset, figname="drive_random.png", num_images=3)
logger.success("Saved example images and masks to 'figures'")

# Simple Encoder-Decoder on PH2

LEARNING_RATE = 0.001
MAX_EPOCHS = 20
loss_fn = bce_loss


encdec_ph2_model = EncDec(input_channels=3, output_channels=1)
optimizer = torch.optim.Adam(encdec_ph2_model.parameters(), lr=LEARNING_RATE)

# This config is used for WANDB
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

# This config is used for WANDB
config= {
    "learning_rate": LEARNING_RATE,
    "architecture": "Simple-Encoder-Decoder",
    "dataset": "DRIVE",
    "epochs": MAX_EPOCHS,
    "loss_fn": "BinaryCrossEntropy",
    "optimizer": "Adam"
}

logger.working_on("Training simple Encoder-Decoder on DRIVE")
train_model(encdec_drive_model, drive_train_loader, drive_val_loader, loss_fn, optimizer, wandb_config=config, num_epochs= MAX_EPOCHS, device=DEVICE)
visualize_predictions(encdec_drive_model, drive_train_loader, DEVICE, figname="drive_predictions.png", num_images=3)
logger.success("Saved examples of predictions for Enc-Dec of DRIVE to 'figures'")
  

# U-Net
print("*" * 100)
print("Training UNet now!")
print("*" * 100)

# Simple Encoder-Decoder on UNet

LEARNING_RATE = 0.001
MAX_EPOCHS = 20
loss_fn = bce_loss

UNetModel = UNet(in_channels=3, num_classes=1)
optimizer = torch.optim.Adam(UNetModel.parameters(), lr=LEARNING_RATE)


#This is config used for WANDB
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

print("just the second UNet")
#encdec_drive_model = EncDec(input_channels=3, output_channels=1) # line is temporary


# Simple UNet on DRIVE
LEARNING_RATE = 0.001
MAX_EPOCHS = 20
loss_fn = focal_loss

UNetModel = UNet(in_channels=3, num_classes=1)
optimizer = torch.optim.Adam(UNetModel.parameters(), lr=LEARNING_RATE)

# This config is used for WANDB
config= {
    "learning_rate": LEARNING_RATE,
    "architecture": "UNet",
    "dataset": "DRIVE",
    "epochs": MAX_EPOCHS,
    "loss_fn": "FocalLoss",
    "optimizer": "Adam"
}

logger.working_on("Training simple UNet on DRIVE")
train_model(UNetModel, drive_train_loader, drive_val_loader, loss_fn, optimizer, wandb_config=config, num_epochs= MAX_EPOCHS, device=DEVICE)
visualize_predictions(encdec_drive_model, drive_train_loader, DEVICE, figname="drive_predictions_UNet_fl.png", num_images=3)
logger.success("Saved examples of predictions for Enc-Dec of DRIVE to 'figures'")


# Evaluation

# Plots
