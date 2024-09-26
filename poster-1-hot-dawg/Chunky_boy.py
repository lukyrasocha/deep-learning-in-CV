import numpy as np
from tqdm.notebook import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import torchvision.transforms as transforms

from utils import set_plot_style, visualize_samples, plot_training_curves, Hotdog_NotHotdog
from models import ChunkyBoy
from smoothgrad import smooth_grad, visualize_saliency_map


def train(model, optimizer, device, train_loader, val_loader, num_epochs=10):
    loss_fun = nn.BCEWithLogitsLoss()

    out_dict = {'train_acc': [],
                'val_acc': [],
                'train_loss': [],
                'val_loss': []}

    for epoch in range(num_epochs):
        model.train()
        train_correct = 0
        train_incorrect = 0
        train_loss = []
        for minibatch_no, (data, target) in enumerate(train_loader):

            data, target = data.to(device), target.to(device).float().unsqueeze(1)
            optimizer.zero_grad()
            output = model(data)
            loss = loss_fun(output, target)
            loss.backward()
            optimizer.step()

            train_loss.append(loss.item())
            predicted = (torch.sigmoid(output) > 0.5).float()
            train_correct += (target == predicted).sum().cpu().item()
            train_incorrect += (target != predicted).sum().cpu().item()

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

        print(f"Epoch {epoch+1}/{num_epochs} - "
              f"Loss train: {train_avg_loss:.3f}\t Val loss: {val_avg_loss:.3f}\t"
              f"Accuracy train: {train_acc*100:.1f}%\t Val: {val_acc*100:.1f}%")
    return out_dict, model


def main():
    set_plot_style()

    # check whether mps cuda or cpu
    if torch.backends.mps.is_built():
        device = torch.device('mps')
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    data_dir = 'data'

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
    full_trainset = Hotdog_NotHotdog(train=True, transform=train_transform, data_path=data_dir)
    testset = Hotdog_NotHotdog(train=False, transform=test_transform, data_path=data_dir)

    validation_split = 0.2  # use 20% of the training data for validation
    total_train = len(full_trainset)
    val_size = int(total_train * validation_split)
    train_size = total_train - val_size

    # Split the dataset
    train_subset, val_subset = random_split(full_trainset, [train_size, val_size])

    # Create DataLoaders
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=0)

    print(f'The number of images in training set is: {len(train_subset)}')
    print(f'The number of images in validation set is: {len(val_subset)}')
    print(f'The number of images in test set is: {len(testset)}')

    # Visualize some training images
    visualize_samples(train_loader)

    # Model setup
    cnn_model = ChunkyBoy().to(device)

    optimizer = torch.optim.Adam(cnn_model.parameters(), lr=0.001)

    print('Training the Chunky Model')
    nn_out_dict, model = train(cnn_model, optimizer, device, train_loader, test_loader, num_epochs=10)
    plot_training_curves(nn_out_dict)

    # Get a random sample from the validation set
    data_iter = iter(val_loader)
    images, labels = next(data_iter)
    index = np.random.randint(len(images))
    sample_image = images[index]
    sample_label = labels[index]

    for i in range(10):
        index = np.random.randint(len(images))
        sample_image = images[index]
        sample_label = labels[index]
        saliency_map = smooth_grad(model, sample_image, sample_label, device)
        visualize_saliency_map(sample_image, saliency_map)


if __name__ == '__main__':
    main()
