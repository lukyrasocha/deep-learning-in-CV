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


def main():
    device = torch.device('mps' if torch.has_mps else 'cpu')  # Fallback to CPU if MPS not available

    data_dir = 'data'

    class Hotdog_NotHotdog(torch.utils.data.Dataset):
        def __init__(self, train, transform, data_path=data_dir):
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
    trainset = Hotdog_NotHotdog(train=True, transform=train_transform)
    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True,
                              num_workers=0)  # Consider num_workers=0 if issues persist
    testset = Hotdog_NotHotdog(train=False, transform=test_transform)
    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=0)

    print(f'The number of images in training set is: {len(trainset)}')
    print(f'The number of images in test set is: {len(testset)}')

    # Visualize some training images
    images, labels = next(iter(train_loader))
    plt.figure(figsize=(20, 10))

    for i in range(21):
        plt.subplot(5, 7, i+1)
        plt.imshow(np.transpose(images[i].numpy(), (1, 2, 0)))
        plt.title(['hotdog', 'not hotdog'][labels[i].item()])
        plt.axis('off')

    plt.show()

    class CNN(nn.Module):
        def __init__(self):
            super(CNN, self).__init__()
            self.convolutional = nn.Sequential(
                nn.Conv2d(in_channels=3, out_channels=8, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(in_channels=8, out_channels=8, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(in_channels=8, out_channels=8, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(in_channels=8, out_channels=8, kernel_size=3, padding=1),
                nn.ReLU()
            )

            self.fully_connected = nn.Sequential(
                nn.Linear(64*64*8, 500),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(500, 10)
            )

        def forward(self, x):
            x = self.convolutional(x)
            x = x.view(x.size(0), -1)
            x = self.fully_connected(x)
            return x

    model = CNN()
    model.to(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    # We define the training as a function so we can easily re-use it.
    def train(model, optimizer, num_epochs=10):
        def loss_fun(output, target):
            return F.cross_entropy(output, target)
        out_dict = {'train_acc': [],
                    'test_acc': [],
                    'train_loss': [],
                    'test_loss': []}

        for epoch in tqdm(range(num_epochs), unit='epoch'):
            model.train()
            # For each epoch
            train_correct = 0
            train_loss = []
            for minibatch_no, (data, target) in tqdm(enumerate(train_loader), total=len(train_loader), leave=False):
                data, target = data.to(device), target.to(device)
                # Zero the gradients computed for each weight
                optimizer.zero_grad()
                # Forward pass your image through the network
                output = model(data)
                # Compute the loss
                loss = loss_fun(output, target)
                # Backward pass through the network
                loss.backward()
                # Update the weights
                optimizer.step()

                train_loss.append(loss.item())
                # Compute how many were correctly classified
                predicted = output.argmax(1)
                train_correct += (target == predicted).sum().cpu().item()
            # Compute the test accuracy
            test_loss = []
            test_correct = 0
            model.eval()
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                with torch.no_grad():
                    output = model(data)
                test_loss.append(loss_fun(output, target).cpu().item())
                predicted = output.argmax(1)
                test_correct += (target == predicted).sum().cpu().item()
            out_dict['train_acc'].append(train_correct / len(trainset))
            out_dict['test_acc'].append(test_correct / len(testset))
            out_dict['train_loss'].append(np.mean(train_loss))
            out_dict['test_loss'].append(np.mean(test_loss))
            print(f"Epoch {epoch+1}/{num_epochs} - "
                  f"Loss train: {np.mean(train_loss):.3f}\t test: {np.mean(test_loss):.3f}\t"
                  f"Accuracy train: {out_dict['train_acc'][-1]*100:.1f}%\t "
                  f"test: {out_dict['test_acc'][-1]*100:.1f}%")
        return out_dict

    out_dict = train(model, optimizer)

    epochs = len(out_dict['train_acc'])

    fig, ax = plt.subplots(figsize=(15, 7), ncols=2)
    ax[0].plot(range(1, epochs + 1), out_dict['train_loss'], label='train_loss')
    ax[0].plot(range(1, epochs + 1), out_dict['test_loss'], label='test_loss')
    ax[0].set_ylabel('Loss')
    ax[0].set_xlabel('Epoch')
    ax[0].legend()

    ax[1].plot(range(1, epochs + 1), out_dict['train_acc'], label='train_acc')
    ax[1].plot(range(1, epochs + 1), out_dict['test_acc'], label='test_acc')
    ax[1].set_ylabel('Accuracy')
    ax[1].set_xlabel('Epoch')
    ax[1].legend()

    plt.show()


if __name__ == '__main__':
    main()
