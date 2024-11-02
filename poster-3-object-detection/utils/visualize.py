from matplotlib import patches
import matplotlib.pyplot as plt
import numpy as np
import random
import os
import torch
from PIL import Image
import matplotlib as mpl
import cv2

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
        
        rect = patches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                 linewidth=box_thickness, edgecolor=color_primary, facecolor='none')
        ax.add_patch(rect)

    
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(f"figures/{figname}", bbox_inches='tight', dpi=300)
    plt.show()