import numpy as np
import matplotlib.pyplot as plt
import os
import glob
from PIL import Image
import torch


class Hotdog_NotHotdog(torch.utils.data.Dataset):
    def __init__(self, train, transform, data_path):
        'Initialization'
        self.transform = transform
        data_path = os.path.join(data_path, 'train' if train else 'test')
        image_classes = [os.path.split(d)[1] for d in glob.glob(data_path + '/*') if os.path.isdir(d)]
        image_classes.sort()
        self.name_to_label = {c: id for id, c in enumerate(image_classes)}
        self.image_paths = glob.glob(data_path + '/*/*.jpg')

    def __len__(self):
        'Returns the total number of samples'
        return len(self.image_paths)

    def __getitem__(self, idx):
        'Generates one sample of data'
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')  # Ensure 3 channels
        c = os.path.split(os.path.split(image_path)[0])[1]
        y = self.name_to_label[c]
        X = self.transform(image)
        return X, y


def set_plot_style():
    plt.rcParams['axes.prop_cycle'] = plt.cycler(color=['#990000', '#2F3EEA', '#030F4F'])
    plt.rcParams['figure.facecolor'] = 'white'
    plt.rcParams['axes.grid'] = True
    plt.rcParams['grid.color'] = '#d3d3d3'


def visualize_samples(train_loader):
    images, labels = next(iter(train_loader))
    plt.figure(figsize=(20, 10))
    color_primary = '#990000'  # University red
    color_secondary = '#2F3EEA'  # University blue

    for i in range(min(4, len(images))):
        plt.subplot(1, 4, i+1)
        plt.imshow(np.transpose(images[i].numpy(), (1, 2, 0)))
        plt.title(['Hotdog', 'Not Hotdog'][labels[i].item()], color=color_primary)
        plt.axis('off')

    plt.suptitle('Sample Training Images', fontsize=24, color=color_secondary)
    plt.tight_layout()
    plt.show()


def plot_training_curves(nn_out):
    epochs = len(nn_out['train_acc'])
    epoch_range = range(1, epochs + 1)

    # Set up university colors
    color_primary = '#990000'  # University red
    color_secondary = '#2F3EEA'  # University blue
    color_accent = '#030F4F'  # Dark blue

    plt.figure(figsize=(18, 10))

    # Plot Loss
    plt.subplot(2, 1, 1)
    plt.plot(epoch_range, nn_out['train_loss'], label='NN Train Loss', color=color_primary, linestyle='-')
    if 'val_loss' in nn_out:
        plt.plot(epoch_range, nn_out['val_loss'], label='NN Val Loss', color=color_primary, linestyle='--')
        plt.title('Training and Validation Loss', fontsize=16, color=color_accent)
    else:
        plt.plot(epoch_range, nn_out['test_loss'], label='NN Test Loss', color=color_primary, linestyle='--')
        plt.title('Training and Test Loss', fontsize=16, color=color_accent)
    plt.title('Training and Validation Loss', fontsize=16, color=color_accent)
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('Loss', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True)

    # Plot Accuracy
    plt.subplot(2, 1, 2)
    plt.plot(epoch_range, [acc * 100 for acc in nn_out['train_acc']],
             label='NN Train Acc', color=color_secondary, linestyle='-')
    if 'val_acc' in nn_out:
        plt.plot(epoch_range, [acc * 100 for acc in nn_out['val_acc']],
                 label='NN Val Acc', color=color_secondary, linestyle='--')
        plt.title('Training and Validation Accuracy', fontsize=16, color=color_accent)
    else:
        plt.plot(epoch_range, [acc * 100 for acc in nn_out['test_acc']],
                 label='NN Test Acc', color=color_secondary, linestyle='--')
        plt.title('Training and Validation Accuracy', fontsize=16, color=color_accent)
    #plt.plot(epoch_range, [acc * 100 for acc in nn_out['val_acc']],
    #         label='NN Val Acc', color=color_secondary, linestyle='--')
    plt.title('Training and Validation Accuracy', fontsize=16, color=color_accent)
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('Accuracy (%)', fontsize=14)
    plt.ylim(0, 100)
    plt.legend(fontsize=12)
    plt.grid(True)

    plt.tight_layout()

    plt.savefig('figures/train_baseline.png')
    plt.show()
