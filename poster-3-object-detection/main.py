import torch.nn as nn
import torch.optim as optim
import os
from torch.utils.data import DataLoader
from torchvision import transforms
from models.models import ResNetTwoHeads
from models.train import train_model
from utils.load_data import Trainingset, ValAndTestDataset, collate_fn, val_test_collate_fn_cropped
from utils.logger import logger

# Paths
blackhole_path = os.getenv('BLACKHOLE')
if not blackhole_path:
    raise EnvironmentError("The $BLACKHOLE environment variable is not set or is empty.")

# Initialize model
model = ResNetTwoHeads().cuda()

# Transformations
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

# Load Training Data
logger.working_on("Loading Train data")
train_dataset = Trainingset(
    image_dir=os.path.join(blackhole_path, 'DLCV/training_data/images'), 
    target_dir=os.path.join(blackhole_path, 'DLCV/training_data/targets'), 
    transform=transform
)
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=4, collate_fn=collate_fn)

# Load Validation Data
logger.working_on("Loading Validation data")
val_dataset = ValAndTestDataset(
    base_dir=os.path.join(blackhole_path,'DLCV'),
    split='val', 
    transform=transform,
    orig_data_path= os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Potholes')
)

assert len(train_dataset) != 0, "Data not loaded correctly"
assert len(val_dataset) != 0, "Data not loaded correctly"

val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, collate_fn=val_test_collate_fn_cropped)


# Loss and Optimizer
criterion_cls = nn.CrossEntropyLoss()
criterion_bbox = nn.MSELoss()  # Mean Squared Error for (tx, ty, tw, th)
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# Training and Validation
logger.working_on("Training model")
train_model(model, train_loader, val_loader, criterion_cls, criterion_bbox, optimizer, num_epochs=10)