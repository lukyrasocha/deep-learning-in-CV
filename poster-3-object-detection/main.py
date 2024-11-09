import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import models, transforms

from models.models import ResNetTwoHeads
from models.train import train_model
from utils.load_data import PotholeTrainDataset, collate_fn



model = ResNetTwoHeads().cuda()
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

dataset = PotholeTrainDataset(images_dir='Potholes/training_data_images', targets_dir='Potholes/training_data_targets', transform=transform)
dataloader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=collate_fn, num_workers = 4)

criterion_cls = nn.CrossEntropyLoss()
criterion_bbox = nn.MSELoss()  # Mean Squared Error for (tx, ty, tw, th)
optimizer = optim.Adam(model.parameters(), lr=1e-4)

train_model(model, dataloader, criterion_cls, criterion_bbox, optimizer, num_epochs=10)