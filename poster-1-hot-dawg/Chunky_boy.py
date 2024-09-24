import numpy as np
from tqdm.notebook import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import os
import glob
import PIL.Image as Image

from utils import set_plot_style, visualize_samples, plot_training_curves, Hotdog_NotHotdog
from models import ChunkyBoy


def train(model, optimizer, device, train_loader, val_loader, num_epochs=10):
    loss_fun = nn.BCEWithLogitsLoss()

    out_dict = {'train_acc': [],
                'val_acc': [],
                'train_loss': [],
                'val_loss': []}

    for epoch in tqdm(range(num_epochs), unit='epoch'):
        model.train()
        train_correct = 0
        train_loss = []
        for minibatch_no, (data, target) in tqdm(enumerate(train_loader), total=len(train_loader), leave=False):

            data, target = data.to(device), target.to(device).float().unsqueeze(1)
            optimizer.zero_grad()
            output = model(data)
            loss = loss_fun(output, target)
            loss.backward()
            optimizer.step()

            train_loss.append(loss.item())
            predicted = (torch.sigmoid(output) > 0.5).float()
            train_correct += (target == predicted).sum().cpu().item()

            # Compute the training accuracy and loss
            train_length = len(train_loader.dataset)
            train_acc = train_correct / train_length
            train_avg_loss = np.mean(train_loss)
            out_dict['train_acc'].append(train_acc)
            out_dict['train_loss'].append(train_avg_loss)

            # Evaluate on validation set
            model.eval()
            val_loss = []
            val_correct = 0
            with torch.no_grad():
                for data, target in val_loader:
                    data, target = data.to(device), target.to(device).float().unsqueeze(1)
                    output = model(data)
                    loss = loss_fun(output, target)
                    val_loss.append(loss.item())
                    predicted = (torch.sigmoid(output) > 0.5).float()
                    val_correct += (target == predicted).sum().cpu().item()

                val_length = len(val_loader.dataset)
                val_acc = val_correct / val_length
                val_avg_loss = np.mean(val_loss)
                out_dict['val_acc'].append(val_acc)
                out_dict['val_loss'].append(val_avg_loss)

                print(f"Epoch {epoch + 1}/{num_epochs} - "
                      f"Loss train: {train_avg_loss:.3f}\t Val loss: {val_avg_loss:.3f}\t"
                      f"Accuracy train: {train_acc * 100:.1f}%\t Val: {val_acc * 100:.1f}%")
    return out_dict

def main():
    set_plot_style()

    # check whether mps cuda or cpu
    if torch.has_mps:
        device = torch.device('mps')
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    data_dir = 'hotdog_nothotdog'


    size = 128
    train_transform = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor()
    ])
    test_transform = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor()
    ])

    batch_size = 64
    trainset = Hotdog_NotHotdog(train=True, transform=train_transform, data_path=data_dir)
    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=0)
    testset = Hotdog_NotHotdog(train=False, transform=test_transform, data_path=data_dir)
    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=0)

    print(f'The number of images in training set is: {len(trainset)}')
    print(f'The number of images in test set is: {len(testset)}')

    # Visualize some training images
    visualize_samples(train_loader)

    # Model setup
    cnn_model = ChunkyBoy().to(device)

    optimizer = torch.optim.SGD(cnn_model.parameters(), lr=0.1)

    print('Training the Chunky Model')
    nn_out_dict = train(cnn_model, optimizer, device, train_loader, test_loader, num_epochs=2)
    plot_training_curves(nn_out_dict)
