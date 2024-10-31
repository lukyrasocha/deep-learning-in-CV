import matplotlib.pyplot as plt
import numpy as np
import random

color_primary = '#990000'  # University red
color_secondary = '#2F3EEA'  # University blue
color_tertiary = '#F6D04D'  # University gold

def visualize_samples(dataloader,figname, num_images=4, class_names=['Not Pothole', 'Pothole'], box_thickness=5):
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

    plt.suptitle('Sample Training Images with Bounding Boxes', fontsize=42, color=color_primary, y=0.81)
    plt.tight_layout()
    plt.savefig(f"figures/{figname}.svg", dpi=300, bbox_inches='tight')
    plt.show()

    # print hello
