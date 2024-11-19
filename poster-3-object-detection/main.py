import argparse
import random
import torch.nn as nn
import torch.optim as optim
import os
from torch.utils.data import DataLoader
from torchvision import transforms
from models.models import ResNetTwoHeads
from models.train import train_model, evaluate_model
from utils.load_data import Trainingset, ValAndTestDataset, collate_fn, val_test_collate_fn_cropped
from utils.logger import logger
from utils.visualize import visualize_predictions, visualize_pred_training_data
import torch
from torch.utils.data import Subset

def main(args):
    # Optional: Set a random seed for reproducibility
    RANDOM_SEED = 42
    random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)

    # Paths
    blackhole_path = os.getenv('BLACKHOLE')
    if not blackhole_path:
        raise EnvironmentError("The $BLACKHOLE environment variable is not set or is empty.")

    # Initialize model
    model = ResNetTwoHeads().cuda()

    # Transformations
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Load Training Data
    logger.working_on("Loading Train data")
    train_dataset = Trainingset(
        image_dir=os.path.join(blackhole_path, 'DLCV/training_data/images'), 
        target_dir=os.path.join(blackhole_path, 'DLCV/training_data/targets'), 
        transform=transform
    )

    # Load Validation Data
    logger.working_on("Loading Validation data")
    val_dataset = ValAndTestDataset(
        base_dir=os.path.join(blackhole_path,'DLCV'),
        split='val', 
        transform=transform,
        orig_data_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Potholes')
    )

    logger.working_on("Loading Test data")
    test_dataset = ValAndTestDataset(
        base_dir=os.path.join(blackhole_path,'DLCV'),
        split='test', 
        transform=transform,
        orig_data_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Potholes')
    )
    assert len(train_dataset) != 0, "Training data not loaded correctly"
    assert len(val_dataset) != 0, "Validation data not loaded correctly"
    assert len(test_dataset) != 0, "Test data not loaded correctly"

    # Handle Subset Selection
    if args.subset_size is not None or args.subset_fraction is not None:
        # Determine subset size
        if args.subset_size is not None:
            subset_size = args.subset_size
            if subset_size > len(train_dataset):
                raise ValueError(f"subset_size {subset_size} exceeds the total number of training samples {len(train_dataset)}")
            subset_size_val = args.subset_size
            if subset_size_val > len(val_dataset):
                raise ValueError(f"subset_size {subset_size_val} exceeds the total number of validation samples {len(val_dataset)}")
        elif args.subset_fraction is not None:
            if not (0.0 < args.subset_fraction <= 1.0):
                raise ValueError("subset_fraction must be between 0 and 1")
            subset_size = int(len(train_dataset) * args.subset_fraction)
            subset_size_val = int(len(val_dataset) * args.subset_fraction)
            if subset_size == 0 or subset_size_val == 0:
                raise ValueError("subset_fraction too small, resulting in zero samples")

        # Generate random indices
        train_indices = list(range(len(train_dataset)))
        val_indices = list(range(len(val_dataset)))
        test_indices = list(range(len(test_dataset)))
        random.shuffle(train_indices)
        random.shuffle(val_indices)
        random.shuffle(test_indices)

        train_subset_indices = train_indices[:subset_size]
        val_subset_indices = val_indices[:subset_size_val]
        test_subset_indices = test_indices[:subset_size_val]

        # Create Subset objects
        train_subset = Subset(train_dataset, train_subset_indices)
        val_subset = Subset(val_dataset, val_subset_indices)
        test_subset = Subset(test_dataset, test_subset_indices)

        logger.info(f"Using subset of size {subset_size} for training and {subset_size_val} for validation.")
    else:
        train_subset = train_dataset
        val_subset = val_dataset
        test_subset = test_dataset
        logger.info("Using the entire dataset for training and validation.")

    # Create DataLoaders
    train_loader = DataLoader(
        train_subset, 
        batch_size=1, 
        shuffle=True, 
        num_workers=4, 
        collate_fn=collate_fn
    )

    val_loader = DataLoader(
        val_subset, 
        batch_size=1, 
        shuffle=False, 
        collate_fn=val_test_collate_fn_cropped
    )

    test_loader = DataLoader(
        test_subset, 
        batch_size=1, 
        shuffle=False, 
        collate_fn=val_test_collate_fn_cropped
    )

    # Loss and Optimizer
    criterion_cls = nn.CrossEntropyLoss()
    criterion_bbox = nn.MSELoss()  # Mean Squared Error for (tx, ty, tw, th)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    # Training and Validation
    logger.working_on("Training model")
    train_model(
        model, train_loader, val_loader, criterion_cls, criterion_bbox, optimizer, 
        num_epochs=args.num_epochs, 
        iou_threshold=args.confidence_threshold, 
        cls_weight=args.cls_weight, 
        reg_weight=args.reg_weight, 
        experiment_name=args.experiment_name
    )

    # Visualize Predictions
    logger.working_on("Visualizing predictions")
    visualize_predictions(
        model=model,
        dataloader=test_loader,
        use_nms=True,  # Set to False to display all proposals
        iou_threshold=args.iou_threshold,  # For NMS, overlapping boxes with 0.3 IoU will get filtered (the better one will stay)
        num_images=args.num_images,
        experiment_name=args.experiment_name
    )
    visualize_pred_training_data(
        model, train_loader, use_nms=True, iou_threshold=args.iou_threshold, 
        num_images=5, experiment_name=args.experiment_name
    )

        # Save the Trained Model
    logger.working_on("Saving model")
    os.makedirs("saved_models", exist_ok=True)
    
    # Save state_dict (recommended)
    state_dict_save_path = f"saved_models/model_state_dict_{args.experiment_name}.pth"
    torch.save(model.state_dict(), state_dict_save_path)
    logger.info(f"Model state_dict saved to {state_dict_save_path}")
    
    # Save the Entire Model (optional)
    model_save_path = f"saved_models/model_full_{args.experiment_name}.pth"
    torch.save(model, model_save_path)
    logger.info(f"Entire model saved to {model_save_path}")


    # Evaluate the Model
    logger.working_on("Evaluating model on Validation split")
    ap, precision, recall = evaluate_model(
        model, 
        val_loader, 
        split='val',
        experiment_name=args.experiment_name, 
        iou_threshold=args.iou_threshold, 
        confidence_threshold=args.confidence_threshold
    )

    print(f"precision: {precision}")
    print(f"recall: {recall}")
    logger.info(f"Experiment {args.experiment_name} - mAP: {ap:.4f}")

    # Evaluate the Model
    logger.working_on("Evaluating model on Test split")
    ap, precision, recall = evaluate_model(
        model, 
        test_loader, 
        split='test',
        experiment_name=args.experiment_name, 
        iou_threshold=args.iou_threshold, 
        confidence_threshold=args.confidence_threshold
    )

    print(f"precision: {precision}")
    print(f"recall: {recall}")
    logger.info(f"Experiment {args.experiment_name} - mAP: {ap:.4f}")

    logger.success("Predictions saved to 'figures/'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate a two-headed ResNet model.")

    # Existing arguments
    parser.add_argument('--experiment_name', type=str, default='experiment1', help='Name of the experiment')
    parser.add_argument('--num_images', type=int, default=5, help='Number of images to visualize')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate for the optimizer')
    parser.add_argument('--num_epochs', type=int, default=4, help='Number of training epochs')
    parser.add_argument('--iou_threshold', type=float, default=0.5, help='IoU threshold for evaluation and NMS')
    parser.add_argument('--confidence_threshold', type=float, default=0.5, help='Confidence threshold for evaluation')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay for the optimizer')
    parser.add_argument('--cls_weight', type=float, default=1.0, help='Weight for classification loss')
    parser.add_argument('--reg_weight', type=float, default=1.0, help='Weight for regression loss')

    # New mutually exclusive arguments for subset selection
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        '--subset_size', 
        type=int, 
        default=None, 
        help='Number of samples to use for training and validation. If specified, a subset of this size will be used.'
    )
    group.add_argument(
        '--subset_fraction', 
        type=float, 
        default=None, 
        help='Fraction of the dataset to use for training and validation (between 0 and 1). If specified, a subset of this fraction will be used.'
    )

    args = parser.parse_args()

    main(args)
