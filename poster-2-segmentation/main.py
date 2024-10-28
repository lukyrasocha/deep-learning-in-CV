import argparse
import torch
import numpy as np
import wandb
import random

from torchvision import transforms
from torch.utils.data import DataLoader

from utils.load_data import load_data
from utils.logger import logger
from utils.transforms import JointTransform
from utils.visualize import display_random_images_and_masks, visualize_predictions
from models.train import train_model
from models.models import EncDec, UNet
from models.losses import bce_loss
from models.metrics import dice_overlap, IoU, accuracy, sensitivity, specificity
from models.evaluation import evaluate_model

def main():
    # Argument parser
    parser = argparse.ArgumentParser(description='Train and evaluate segmentation models on datasets.')
    parser.add_argument('--model', type=str, default='unet', choices=['unet', 'encdec'],
                        help='Model to use: unet or encdec')
    parser.add_argument('--data', type=str, default='ph2', choices=['ph2', 'drive'],
                        help='Dataset to use: ph2 or drive')

    parser.add_argument('--loss_fn', type=str, default='bce', choices=['bce','focal'])
    parser.add_argument('--epochs', type=int, default=2, help='Number of epochs to train')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--crop_size', type=int, default=256, help='Crop size for training')
    parser.add_argument('--resize', type=int, default=None, help='Resize size (set to None for no resizing)')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training')
    parser.add_argument('--padding', type=int, default=0, help='Padding for UNet (0 means no padding)')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use for training (cuda or cpu)')
    parser.add_argument('--visualize', action='store_true', help='Whether to visualize predictions')
    parser.add_argument('--jobid', type=str, default=f"job-{random.randint(1,10**8)}")
    args = parser.parse_args()

    DEVICE = torch.device(args.device)
    logger.info(f"Running on {DEVICE}")

    # Initialize Weights & Biases (optional)
    wandb.init(project='project2-segmentation', config=args)

    wandb.init(
        project="project2-segmentation",
        name=args.jobid,
        config= args
    )

    # Transformations
    RESIZE = (args.resize, args.resize) if args.resize else None
    CROP_SIZE = (args.crop_size, args.crop_size) if args.crop_size else None

    transform_train = JointTransform(crop_size=CROP_SIZE, resize=RESIZE)
    transform_val_test = transforms.Compose([
        transforms.ToTensor(),
    ])

    # Load data
    logger.working_on(f"Loading data for {args.data.upper()}")
    train_dataset = load_data(args.data, split='train', transform=transform_train, crop=True)
    val_dataset = load_data(args.data, split='val', transform=transform_train, crop=True)
    test_dataset = load_data(args.data, split='test', transform=transform_val_test)

    # Check mask values
    image, mask = train_dataset[1]
    print('Image shape:', image.shape)
    print('Mask shape:', mask.shape)
    assert len(np.unique(mask.numpy()[0])) <= 2, "Mask needs to have binary values (0,1)"

    # Data loaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    logger.success("Data loaded")

    # Display some images
    display_random_images_and_masks(train_dataset, figname=f"{args.data}_random.png", num_images=3)
    logger.success(f"Saved example images and masks for {args.data.upper()} to 'figures'")

    # Model selection
    if args.model == 'encdec':
        model = EncDec(input_channels=3, output_channels=1)
        architecture = "Simple-Encoder-Decoder"
    elif args.model == 'unet':
        model = UNet(in_channels=3, num_classes=1, padding=args.padding)
        architecture = "UNet"

    model = model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    if args.loss_fn == 'bce':
        loss_fn = bce_loss
    else:
        raise "Not implemented"

    logger.working_on(f"Training {architecture} on {args.data.upper()}")
    train_model(model, train_loader, val_loader, loss_fn, optimizer, num_epochs=args.epochs, device=DEVICE)

    if args.visualize:
        visualize_predictions(model, train_loader, DEVICE,
                              figname=f"{architecture}_{args.data}_predictions.png", num_images=5)
        logger.success(f"Saved examples of predictions for {architecture} on {args.data.upper()} to 'figures'")

    # Evaluation
    # Reload datasets without cropping for evaluation
    train_dataset = load_data(args.data, split='train', transform=transform_val_test, crop=False)
    val_dataset = load_data(args.data, split='val', transform=transform_val_test, crop=False)
    test_dataset = load_data(args.data, split='test', transform=transform_val_test, crop=False)

    # Data loaders for evaluation
    eval_train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False)
    eval_val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    eval_test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # Evaluation metrics
    metrics = [dice_overlap, IoU, accuracy, sensitivity, specificity]

    eval_config = {
        "architecture": architecture,
        "dataset": args.data.upper(),
        "loss_function": "BinaryCrossEntropy",
        "patch_size": args.crop_size,
    }

    evaluate_model(model, eval_val_loader, DEVICE, metrics, dataset_name=args.data.upper(),
                   patch_size=args.crop_size, add_edge=True)

    wandb.finish()

if __name__ == "__main__":
    main()