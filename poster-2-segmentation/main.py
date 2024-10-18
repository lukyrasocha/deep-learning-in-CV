import torch

from torchvision import transforms

from utils.load_data import load_data
from utils.logger import logger

# Load data
logger.working_on("Loading data for PH2")

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

ph2_train_dataset = load_data('ph2', split='train', transform=transform)
ph2_val_dataset = load_data('ph2', split='val', transform=transform)
ph2_test_dataset = load_data('ph2', split='test', transform=transform)

print('PH2 Train Dataset size:', len(ph2_train_dataset))
print('PH2 Validation Dataset size:', len(ph2_val_dataset))
print('PH2 Test Dataset size:', len(ph2_test_dataset))

image, mask = ph2_train_dataset[1]
print('Image shape:', image.shape)
print('Mask shape:', mask.shape)

# Data loaders
ph2_train_loader = torch.utils.data.DataLoader(ph2_train_dataset, batch_size=16, shuffle=True)
ph2_val_loader = torch.utils.data.DataLoader(ph2_val_dataset, batch_size=16, shuffle=False)
ph2_test_loader = torch.utils.data.DataLoader(ph2_test_dataset, batch_size=16, shuffle=False)

logger.working_on("Loading data for DRIVE")

drive_train_dataset = load_data('drive', split='train', transform=transform)
drive_val_dataset = load_data('drive', split='val', transform=transform)
drive_test_dataset = load_data('drive', split='test', transform=transform)

print('DRIVE Train Dataset size:', len(drive_train_dataset))
print('DRIVE Validation Dataset size:', len(drive_val_dataset))
print('DRIVE Test Dataset size:', len(drive_test_dataset))

# Data loaders
drive_train_loader = torch.utils.data.DataLoader(drive_train_dataset, batch_size=4, shuffle=True)
drive_val_loader = torch.utils.data.DataLoader(drive_val_dataset, batch_size=4, shuffle=False)
drive_test_loader = torch.utils.data.DataLoader(drive_test_dataset, batch_size=4, shuffle=False)

image, mask = drive_train_dataset[1]
print('Image shape:', image.shape)
print('Mask shape:', mask.shape)

logger.success("Data loaded")

# Simple Encoder-Decoder

# U-Net

# Evaluation

# Plots