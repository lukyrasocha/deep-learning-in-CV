import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import random
import os
import torch
from PIL import Image
import matplotlib as mpl
import cv2
from utils.metrics import non_max_suppression

color_primary = '#990000'  # University red
color_secondary = '#2F3EEA'  # University blue
color_tertiary = '#F6D04D'  # University gold

mpl.rcParams.update(mpl.rcParamsDefault)
plt.rcParams.update({'text.usetex': True})


# Set global font size for title, x-label, and y-label
plt.rcParams['axes.titlesize'] = 20
plt.rcParams['axes.labelsize'] = 16

# Set global font size for x and y tick labels
plt.rcParams['xtick.labelsize'] = 16
plt.rcParams['ytick.labelsize'] = 16

# Set global font size for the legend
plt.rcParams['legend.fontsize'] = 16

# Set global font size for the figure title
plt.rcParams['figure.titlesize'] = 42

def visualize_samples(dataloader,figname, num_images=4, box_thickness=5): # add num_casses class_names=['Background', 'Pothole'] if you want to vislize the labels 
    images, targets = next(iter(dataloader))
    plt.figure(figsize=(20, 10))

    # set seed to get always the same images
    random.seed(42)

    for i in range(min(num_images, len(images))):
        image = images[i]
        target = targets[i]
        boxes = target['boxes']
        labels = target['labels']

        # Convert image to numpy array and transpose to H x W x C
        image_np = image.permute(1, 2, 0).numpy()
        image_np = (image_np * 255).astype(np.uint8)  # Scale back to [0, 255] and convert to uint8

        plt.subplot(1, num_images, i + 1)
        plt.imshow(image_np)
        ax = plt.gca()

        # Plot each bounding box
        for box, label in zip(boxes, labels):
            xmin, ymin, xmax, ymax = box
            width = xmax - xmin
            height = ymax - ymin

            # Create a Rectangle patch
            rect = plt.Rectangle((xmin, ymin), width, height, linewidth=box_thickness, edgecolor=color_primary, facecolor='none')
            ax.add_patch(rect)

            # Add label
            #ax.text(xmin, ymin - 10, class_names[label.item()], color=color_tertiary, fontsize=16, weight='bold')

        plt.axis('off')

        # Build the path to save the figure in the parent 'figures' directory
    plt.suptitle('Sample Training Images with Bounding Boxes', color='red', y=0.81)
    plt.tight_layout()

    # Build the path to save the figure in the parent 'figures' directory
    current_dir = os.path.dirname(__file__)
    parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
    figures_dir = os.path.join(parent_dir, 'figures')

    # Ensure the 'figures' directory exists
    os.makedirs(figures_dir, exist_ok=True)

    # Full path to save the figure
    fig_path = os.path.join(figures_dir, f"{figname}.svg")

    # Save the figure
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.show()


def visualize_proposals(image, proposals, num_proposals=500, box_thickness=1, figname='proposals.png'):
    """
    Visualizes the top N proposals on the image.

    Args:
        image (Tensor, numpy array, or PIL Image): The image on which to draw proposals.
        proposals (list): A list of TensorDict objects containing bounding box coordinates.
        num_proposals (int): The number of proposals to visualize.
    """
    # Convert image to PIL Image if it's a tensor or numpy array
    if isinstance(image, torch.Tensor):
        # Scale image to 0-255 if necessary
        if image.max() <= 1:
            image = image * 255
        image = image.byte().permute(1, 2, 0).cpu().numpy()  # Convert to HxWxC and numpy array
        image = Image.fromarray(image)
    elif isinstance(image, np.ndarray):
        # Ensure image is in RGB format if it's BGR
        if image.shape[-1] == 3:  # Check if image has 3 channels
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)

    # Set up plot
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    plt.suptitle(f'Image with a {num_proposals} proposals', color='red')
    ax = plt.gca()
    
    # Draw bounding boxes
    for proposal in proposals[:num_proposals]:
        xmin = proposal['xmin'].item()
        ymin = proposal['ymin'].item()
        xmax = proposal['xmax'].item()
        ymax = proposal['ymax'].item()
        
        rect = patches.Rectangle((xmin, ymin), abs(xmax - xmin), abs(ymax - ymin),
                                 linewidth=box_thickness, edgecolor=color_primary, facecolor='none')
        ax.add_patch(rect)

    
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(f"../figures/{figname}", bbox_inches='tight', dpi=300)
    plt.show()


def visualize_proposal(image_proposal, target_proposal, box_thickness=1, figname='proposals.png'):

    # Convert image to PIL Image if it's a tensor or numpy array
    if isinstance(image_proposal, torch.Tensor):
        # Scale image to 0-255 if necessary
        if image_proposal.max() <= 1:
            image_proposal = image_proposal * 255
        image_proposal = image_proposal.byte().permute(1, 2, 0).cpu().numpy()  # Convert to HxWxC and numpy array
        image_proposal = Image.fromarray(image_proposal)
    elif isinstance(image_proposal, np.ndarray):
        # Ensure image is in RGB format if it's BGR
        if image_proposal.shape[-1] == 3:  # Check if image has 3 channels
            image_proposal = cv2.cvtColor(image_proposal, cv2.COLOR_BGR2RGB)
        image_proposal = Image.fromarray(image_proposal)

    # Set up plot
    plt.figure(figsize=(10, 10))
    plt.imshow(image_proposal)
    ax = plt.gca()
    

    xmin = target_proposal['gt_bbox_xmin'].item()
    ymin = target_proposal['gt_bbox_ymin'].item()
    xmax = target_proposal['gt_bbox_xmax'].item()
    ymax = target_proposal['gt_bbox_ymax'].item()
    
    rect = patches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,linewidth=box_thickness, edgecolor=color_primary, facecolor='none')
    ax.add_patch(rect)

    
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(f"../figures/{figname}", bbox_inches='tight', dpi=300)
    plt.show()

def visualize_predictions(model, dataloader, use_nms=True, iou_threshold=0.3, num_images=5, experiment_name='experiment'):
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    with torch.no_grad():
        for idx, (original_images, proposal_images_list, coords, image_ids, ground_truths) in enumerate(dataloader):
            if idx >= num_images:
                break
            
            original_image = original_images[0]  # Single image in batch
            proposals = coords[0]
            
            # Prepare proposals for the model
            proposal_images = torch.stack(proposal_images_list[0]).to(device)

            # Get predictions
            outputs_cls, outputs_bbox_transforms, cls_probs = model.predict(proposal_images)

            print(f"Image ID: {image_ids[0]}")
            print(f"Number of Proposals: {len(proposals)}")
            #print(f"Predicted Class Probabilities: {cls_probs.tolist()}")

            # Prepare predictions
            predictions = []
            for i, proposal in enumerate(proposals):
                pred_prob = cls_probs[i, 1].item()  # Probability of being a pothole
                if pred_prob >= 0.5:  # Filter low-confidence predictions
                    predictions.append({
                        "pre_bbox_xmin": proposal['xmin'].item(),
                        "pre_bbox_ymin": proposal['ymin'].item(),
                        "pre_bbox_xmax": proposal['xmax'].item(),
                        "pre_bbox_ymax": proposal['ymax'].item(),
                        "pre_class": pred_prob
                    })

            print(f"Predictions of potholes before NMS: {len(predictions)}")

            # Apply Non-Max Suppression if required
            if use_nms:
                predictions = non_max_suppression(predictions, iou_threshold=iou_threshold)
                print(f"Predictions of potholes after NMS: {len(predictions)}")

            
            # Plot the original image with predictions
            fig, ax = plt.subplots(1, figsize=(10, 8))
            ax.imshow(original_image)
            ax.set_title(f"Predictions for Image ID: {image_ids[0]}")
            ax.axis('off')

            # Plot predictions
            for pred in predictions:
                rect = patches.Rectangle(
                    (pred['pre_bbox_xmin'], pred['pre_bbox_ymin']),
                    pred['pre_bbox_xmax'] - pred['pre_bbox_xmin'],
                    pred['pre_bbox_ymax'] - pred['pre_bbox_ymin'],
                    linewidth=2,
                    edgecolor='r',
                    facecolor='none'
                )
                ax.add_patch(rect)
                ax.text(
                    pred['pre_bbox_xmin'], pred['pre_bbox_ymin'] - 10,
                    f"Conf: {pred['pre_class']:.2f}",
                    color='white', fontsize=8,
                    bbox=dict(facecolor='red', alpha=0.5)
                )

            plt.axis('off')
            plt.tight_layout()
            plt.show()

            os.makedirs(f'figures/png/{experiment_name}', exist_ok=True)
            os.makedirs(f'figures/svg/{experiment_name}', exist_ok=True)

            # Save the figure
            nms = "true" if use_nms else "false"
            plt.savefig(f"figures/png/{experiment_name}/predictions_image_{experiment_name}_{image_ids[0]}_nms_{nms}.png", bbox_inches='tight', dpi=300)
            plt.savefig(f"figures/svg/{experiment_name}/predictions_image_{experiment_name}_{image_ids[0]}_nms_{nms}.svg", bbox_inches='tight', dpi=300)

def visualize_pred_training_data(model, train_loader, use_nms=True, iou_threshold=0.3, num_images=5, experiment_name='experiment'):
    """
    Visualize predictions and ground truth for training data.

    Args:
        model: The trained model.
        train_loader: DataLoader for the training data.
        use_nms: Whether to apply non-max suppression.
        iou_threshold: IoU threshold for NMS.
        num_images: Number of images to visualize.
        experiment_name: Name of the experiment (used for saving figures).
    """
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    with torch.no_grad():
        for idx, (images, targets, indices) in enumerate(train_loader):
            if idx >= num_images:
                break
            
            images = images.to(device)
            outputs_cls, outputs_bbox_transforms, cls_probs = model.predict(images)

            print(f"Image Index: {indices}")
            print(f"Number of Proposals: {len(targets)}")

            # Prepare ground truth and predictions
            ground_truths = []
            predictions = []
            ground_truths = []

            for target in targets:
                # Only include positive proposals
                if target["label"].item() == 1:
                    gt_bbox = {
                        "xmin": target["gt_bbox_xmin"].item(),
                        "ymin": target["gt_bbox_ymin"].item(),
                        "xmax": target["gt_bbox_xmax"].item(),
                        "ymax": target["gt_bbox_ymax"].item(),
                    }
                    ground_truths.append(gt_bbox)

            # Extract predictions
            for i, target in enumerate(targets):
                pred_prob = cls_probs[i, 1].item()  # Probability of being a pothole
                if pred_prob >= 0.5:  # Filter low-confidence predictions
                    pred_bbox = {
                        "pre_bbox_xmin": target['image_xmin'].item(),
                        "pre_bbox_ymin": target['image_ymin'].item(),
                        "pre_bbox_xmax": target['image_xmax'].item(),
                        "pre_bbox_ymax": target['image_ymax'].item(),
                        "pre_class": pred_prob
                    }
                    predictions.append(pred_bbox)

            print(f"Ground Truth Boxes: {len(ground_truths)}")
            print(f"Predictions Before NMS: {len(predictions)}")

            # Apply Non-Max Suppression if required
            if use_nms:
                predictions = non_max_suppression(predictions, iou_threshold=iou_threshold)
                print(f"Predictions After NMS: {len(predictions)}")

            # Visualize the original image, predictions, and ground truth
            fig, ax = plt.subplots(1, figsize=(10, 8))
            ax.imshow(images[0].permute(1, 2, 0).cpu().numpy())
            ax.set_title(f"Training Data Visualization - Index: {indices[0]}")
            ax.axis('off')

            # Plot ground truth boxes
            for gt in ground_truths:
                rect = patches.Rectangle(
                    (gt["xmin"], gt["ymin"]),
                    gt["xmax"] - gt["xmin"],
                    gt["ymax"] - gt["ymin"],
                    linewidth=2,
                    edgecolor='g',
                    facecolor='none'
                )
                ax.add_patch(rect)

            # Plot predicted boxes
            for pred in predictions:
                rect = patches.Rectangle(
                    (pred["pre_bbox_xmin"], pred["pre_bbox_ymin"]),
                    pred["pre_bbox_xmax"] - pred["pre_bbox_xmin"],
                    pred["pre_bbox_ymax"] - pred["pre_bbox_ymin"],
                    linewidth=2,
                    edgecolor='r',
                    facecolor='none'
                )
                ax.add_patch(rect)
                ax.text(
                    pred["pre_bbox_xmin"], pred["pre_bbox_ymin"] - 10,
                    f"Conf: {pred['pre_class']:.2f}",
                    color='white', fontsize=8,
                    bbox=dict(facecolor='red', alpha=0.5)
                )

            plt.axis('off')
            plt.tight_layout()
            plt.show()

            os.makedirs(f'figures/png/{experiment_name}', exist_ok=True)
            os.makedirs(f'figures/svg/{experiment_name}', exist_ok=True)
            # Save the visualization
            nms = "true" if use_nms else "false"
            plt.savefig(f"figures/png/{experiment_name}/training_visualization_{experiment_name}_{indices[0]}_nms_{nms}.png", bbox_inches='tight', dpi=300)
            plt.savefig(f"figures/svg/{experiment_name}/training_visualization_{experiment_name}_{indices[0]}_nms_{nms}.svg", bbox_inches='tight', dpi=300)


def visualize_pred_training_data(
    model, 
    train_loader, 
    use_nms=True, 
    iou_threshold=0.3, 
    num_images=5, 
    experiment_name='experiment', 
    image_dir='Potholes/annotated-images'
):
    """
    Visualize predictions and ground truth for training data with the original image.

    Args:
        model: The trained model.
        train_loader: DataLoader for the training data.
        use_nms: Whether to apply non-max suppression.
        iou_threshold: IoU threshold for NMS.
        num_images: Number of images to visualize.
        experiment_name: Name of the experiment (used for saving figures).
        image_dir: Directory containing the original images.
    """
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    with torch.no_grad():
        for idx, (images, targets, indices) in enumerate(train_loader):
            if idx >= num_images:
                break

            # Retrieve the corresponding original image name
            original_image_name = targets[0]['original_image_name']  # Assuming all targets share the same image
            image_path = os.path.join(image_dir, f"{original_image_name}.jpg")

            if not os.path.exists(image_path):
                print(f"Image {image_path} not found. Skipping...")
                continue

            # Load the original image
            with Image.open(image_path) as img:
                original_image = img.convert("RGB")

            # Prepare predictions
            outputs_cls, outputs_bbox_transforms, cls_probs = model.predict(images.to(device))
            print(f"Image Index: {indices}")
            print(f"Number of Proposals: {len(targets)}")

            ground_truths = []
            predictions = []

            # Collect ground truth bounding boxes
            for target in targets:
                if target["label"].item() == 1:  # Include only positive proposals
                    gt_bbox = {
                        "xmin": target["gt_bbox_xmin"].item(),
                        "ymin": target["gt_bbox_ymin"].item(),
                        "xmax": target["gt_bbox_xmax"].item(),
                        "ymax": target["gt_bbox_ymax"].item(),
                    }
                    ground_truths.append(gt_bbox)

            # Collect predictions
            for i, target in enumerate(targets):
                pred_prob = cls_probs[i, 1].item()  # Probability of being a pothole
                if pred_prob >= 0.5:  # Filter low-confidence predictions
                    pred_bbox = {
                        "pre_bbox_xmin": target['image_xmin'].item(),
                        "pre_bbox_ymin": target['image_ymin'].item(),
                        "pre_bbox_xmax": target['image_xmax'].item(),
                        "pre_bbox_ymax": target['image_ymax'].item(),
                        "pre_class": pred_prob
                    }
                    predictions.append(pred_bbox)

            print(f"Ground Truth Boxes: {len(ground_truths)}")
            print(f"Predictions Before NMS: {len(predictions)}")

            # Apply Non-Max Suppression if required
            if use_nms:
                predictions = non_max_suppression(predictions, iou_threshold=iou_threshold)
                print(f"Predictions After NMS: {len(predictions)}")

            # Visualize the original image, predictions, and ground truth
            fig, ax = plt.subplots(1, figsize=(10, 8))
            ax.imshow(original_image)
            ax.set_title(f"Training Data Visualization - {original_image_name}")
            ax.axis('off')

            # Plot ground truth boxes in green
            for gt in ground_truths:
                rect = patches.Rectangle(
                    (gt["xmin"], gt["ymin"]),
                    gt["xmax"] - gt["xmin"],
                    gt["ymax"] - gt["ymin"],
                    linewidth=2,
                    edgecolor='g',
                    facecolor='none',
                    label="Ground Truth"
                )
                ax.add_patch(rect)

            # Plot predicted boxes in red
            for pred in predictions:
                rect = patches.Rectangle(
                    (pred["pre_bbox_xmin"], pred["pre_bbox_ymin"]),
                    pred["pre_bbox_xmax"] - pred["pre_bbox_xmin"],
                    pred["pre_bbox_ymax"] - pred["pre_bbox_ymin"],
                    linewidth=2,
                    edgecolor='r',
                    facecolor='none',
                    label="Prediction"
                )
                ax.add_patch(rect)
                ax.text(
                    pred["pre_bbox_xmin"], pred["pre_bbox_ymin"] - 10,
                    f"Conf: {pred['pre_class']:.2f}",
                    color="white",
                    fontsize=8,
                    bbox=dict(facecolor="red", alpha=0.5)
                )

            plt.axis("off")
            plt.tight_layout()
            # Save the visualization
            os.makedirs(f'figures/png/{experiment_name}', exist_ok=True)
            os.makedirs(f'figures/svg/{experiment_name}', exist_ok=True)
            nms = "true" if use_nms else "false"
            plt.savefig(f"figures/png/{experiment_name}/training_visualization_{experiment_name}_{original_image_name}_nms_{nms}.png", bbox_inches="tight", dpi=300)
            plt.savefig(f"figures/svg/{experiment_name}/training_visualization_{experiment_name}_{original_image_name}_nms_{nms}.svg", bbox_inches="tight", dpi=300)