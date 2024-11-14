import torch
import torch.nn as nn
import torch.optim as optim
import os

from torch.utils.data import DataLoader
from torchvision import models, transforms
from models.models import ResNetTwoHeads
from models.train import train_model
from utils.load_data import Trainingset

blackhole_path = os.getenv('BLACKHOLE')
if not blackhole_path:
    raise EnvironmentError("The $BLACKHOLE environment variable is not set or is empty.")


model = ResNetTwoHeads().cuda()
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

print(os.path.join(blackhole_path,'DLCV/Potholes/training_data_images'))

dataset = Trainingset(image_path=os.path.join(blackhole_path,'DLCV/Potholes/training_data_images'), target_path=os.path.join(blackhole_path, 'DLCV/Potholes/training_data_targets'))
dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers = 4)

criterion_cls = nn.CrossEntropyLoss()
criterion_bbox = nn.MSELoss()  # Mean Squared Error for (tx, ty, tw, th)
optimizer = optim.Adam(model.parameters(), lr=1e-4)

train_model(model, dataloader, criterion_cls, criterion_bbox, optimizer, num_epochs=10)