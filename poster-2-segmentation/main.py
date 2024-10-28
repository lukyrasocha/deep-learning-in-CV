import argparse
import torch
import numpy as np
import wandb
import random

from torchvision import transforms
from torch.utils.data import DataLoader

from utils.load_data import load_data
from utils.logger import logger
from utils.transforms import JointTransform, JointTransform_weak
from utils.visualize import display_random_images_and_masks, visualize_predictions, display_random_images_and_weak_supervision_masks, visualize_weak_supervision_predictions
from utils.helper import compute_pos_weight
from models.train import train_model, train_model_weak
from models.models import EncDec, UNet
from models.losses import bce_loss, masked_bce_loss, weighted_bce_loss, focal_loss
from models.metrics import dice_overlap, IoU, accuracy, sensitivity, specificity
from models.evaluation import evaluate_model

def main():
    # Argument parser
    parser = argparse.ArgumentParser(description='Train and evaluate segmentation models on datasets.')
    parser.add_argument('--model', type=str, default='unet', choices=['unet', 'encdec'],
                        help='Model to use: unet or encdec')
    parser.add_argument('--data', type=str, default='ph2', choices=['ph2', 'drive'],
                        help='Dataset to use: ph2 or drive')

    parser.add_argument('--loss_fn', type=str, default='bce', choices=['bce','focal', 'masked_bce', 'weighted_bce'])
    parser.add_argument('--epochs', type=int, default=2, help='Number of epochs to train')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--crop_size', type=int, default=256, help='Crop size for training')
    parser.add_argument('--resize', type=int, default=None, help='Resize size (set to None for no resizing)')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training')
    parser.add_argument('--padding', type=int, default=0, help='Padding for UNet (0 means no padding)')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use for training (cuda or cpu)')
    parser.add_argument('--weak', action='store_true', help='Weak supervision')
    parser.add_argument('--visualize', action='store_true', help='Whether to visualize predictions')
    parser.add_argument('--jobid', type=str, default=f"job-{random.randint(1,10**8)}")
    parser.add_argument('--num_clicks', type=int, default=15)
    parser.add_argument('--sampling_strategy', type=str, default="random")
    
    args = parser.parse_args()

    SEED = 42
    DEVICE = torch.device(args.device)
    logger.info(f"Running on {DEVICE}")
    logger.info(f"JOB ID: {args.jobid}")

    wandb.init(
        project="project2-segmentation",
        name=args.jobid,
        config= args
    )

    # Transformations
    RESIZE = (args.resize, args.resize) if args.resize else None
    CROP_SIZE = (args.crop_size, args.crop_size) if args.crop_size else None

    if args.data == "ph2":
        mean =  torch.tensor([0.7475, 0.5721, 0.4836])
        std = torch.tensor([0.2004, 0.1972, 0.2023])
    elif args.data == "drive":
        mean =  torch.tensor([0.4820, 0.2620, 0.1546])
        std = torch.tensor([0.3359, 0.1838, 0.1030])

    if args.weak:
        transform_train = JointTransform_weak(crop_size=CROP_SIZE, resize=RESIZE, mean=mean, std=std)
    else:
        transform_train = JointTransform(crop_size=CROP_SIZE, resize=RESIZE, mean = mean, std=std)


    transform_val_test = transforms.Compose([
        transforms.ToTensor(),
        #transforms.Resize((400, 400)),
        transforms.Normalize(mean=mean, std=std)
    ])

    # Load data
    logger.working_on(f"Loading data for {args.data.upper()}")
    if args.weak:
        train_dataset = load_data('ph2_weak_supervision', split='train',transform=transform_train,num_clicks=args.num_clicks, radius=10, crop=True,seed=SEED,
            return_ground_truth=False, sampling=args.sampling_strategy)

        val_dataset = load_data(
            'ph2_weak_supervision',split='val',transform=transform_train,num_clicks=args.num_clicks,radius=10,crop=True,seed=SEED, return_ground_truth=True, sampling=args.sampling_strategy)

        test_dataset = load_data('ph2_weak_supervision',split='test',transform=transform_val_test,return_ground_truth=True)

        image, mask = train_dataset[1]
        print('Image shape:', image.shape)
        print('Mask shape:', mask.shape)
        assert len(np.unique(mask.numpy()[0])) <= 3, "mask needs to have binary values (0,1 and nan)"

    else:
        train_dataset = load_data(args.data, split='train', transform=transform_train, crop=True)
        val_dataset = load_data(args.data, split='val', transform=transform_train, crop=True)
        test_dataset = load_data(args.data, split='test', transform=transform_val_test)
        # Check mask values
        image, mask = train_dataset[1]
        print('Image shape:', image.shape)
        print('Mask shape:', mask.shape)
        assert len(np.unique(mask.numpy()[0])) <= 2, "Mask needs to have binary values (0,1)"

    # Data loaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers = 4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers = 4)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers = 4)

    logger.success("Data loaded")

    # Display some images
    if args.weak:
        display_random_images_and_weak_supervision_masks(train_dataset, figname=f"{args.jobid}-weak_supervision_random.png", num_images=3)
        logger.success(f"Saved example images and masks for weak supervision to 'figures'")

    else:
        display_random_images_and_masks(train_dataset, figname=f"{args.jobid}-{args.data}_random.png", num_images=3)
        logger.success(f"Saved example images and masks for {args.data.upper()} to 'figures'")

    # Model selection
    if args.weak:
        assert args.loss_fn == "masked_bce", "For weak supervision must be masked loss"
        assert args.data == "ph2", "For weak supervision we must have ph2"

    if args.model == 'encdec':
        model = EncDec(input_channels=3, output_channels=1, padding = args.padding)
        architecture = "Simple-Encoder-Decoder"
        if args.weak:
            architecture = "Simple-Encoder-Decoder-weak"

    elif args.model == 'unet':
        model = UNet(in_channels=3, num_classes=1, padding=args.padding)
        architecture = "UNet"
        if args.weak:
            architecture = "UNet-weak"


    model = model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    if args.loss_fn == 'bce':
        loss_fn = bce_loss
    elif args.loss_fn == 'masked_bce':
        loss_fn = masked_bce_loss
    elif args.loss_fn == 'weighted_bce':
        pos_weight = compute_pos_weight(train_dataset).to(DEVICE)
        logger.info(f"Computed pos_weight: {pos_weight.item()}")
        loss_fn = lambda y_pred, y_real: weighted_bce_loss(y_pred, y_real, pos_weight)

    elif args.loss_fn == 'focal':
        loss_fn = focal_loss
    else:
        raise "Not implemented"

    logger.working_on(f"Training {architecture} on {args.data.upper()}")

    if args.weak:
        train_model_weak(model, train_loader, val_loader, loss_fn, optimizer, num_epochs=args.epochs, device=DEVICE)

    else:
        train_model(model, train_loader, val_loader, loss_fn, optimizer, num_epochs=args.epochs, device=DEVICE)

    if args.visualize:
        if args.weak:
            visualize_weak_supervision_predictions(model, train_loader, DEVICE, figname=f"{args.jobid}-{architecture}-weak_supervision_predictions.png", num_images=5, CLICKS=args.num_clicks, SAMPLIG=args.sampling_strategy)
        
        else:
            visualize_predictions(model, train_loader, DEVICE,
                                figname=f"{args.jobid}-{architecture}-{args.data}_predictions.png", num_images=5)

        logger.success(f"Saved examples of predictions for {architecture} on {args.data.upper()} to 'figures'")

    # Evaluation

    torch.save(model, f"saved_models/{args.jobid}-{architecture}-model.pt")

    logger.working_on(f"Evaluating {architecture}...")

    # Reload datasets without cropping for evaluation
    train_dataset = load_data(args.data, split='train', transform=transform_val_test, crop=False)
    val_dataset = load_data(args.data, split='val', transform=transform_val_test, crop=False)
    test_dataset = load_data(args.data, split='test', transform=transform_val_test, crop=False)

    # Data loaders for evaluation
    eval_train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False, num_workers = 4)
    eval_val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers = 4)
    eval_test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers = 4)

    # Evaluation metrics
    metrics = [dice_overlap, IoU, accuracy, sensitivity, specificity]

    if args.padding == 0:
        add_edge = True
    else:
        add_edge = False

    evaluate_model(model, eval_train_loader, DEVICE, metrics, dataset_name=args.data.upper(),
                   patch_size=args.crop_size, name = "train", add_edge=add_edge)

    evaluate_model(model, eval_val_loader, DEVICE, metrics, dataset_name=args.data.upper(),
                   patch_size=args.crop_size, name = "val", add_edge=add_edge)

    evaluate_model(model, eval_test_loader, DEVICE, metrics, dataset_name=args.data.upper(),
                   patch_size=args.crop_size, name = "test", add_edge=add_edge)
    logger.success("Model evaluated and results logged to wandb")

    wandb.finish()

if __name__ == "__main__":
    main()