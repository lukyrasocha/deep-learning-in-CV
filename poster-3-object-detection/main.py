import argparse

from utils.load_data import load_data, custom_collate_fn, Potholes
from torchvision import transforms
from torch.utils.data import DataLoader

def main():
    # Argument parser
    parser = argparse.ArgumentParser(description='Train and evaluate object detection models on datasets.')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training')

    args = parser.parse_args()

    SEED = 42

    #Transformation
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])

    #Get the train, val and test set. The train and test set is given (80% and 20%). The code below set the val to be 20
    train_dataset, val_dataset, test_dataset = load_data(val_percent=20, seed=SEED, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, collate_fn=custom_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers = 4, collate_fn=custom_collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers = 4, collate_fn=custom_collate_fn)


if '__main__':
    main()