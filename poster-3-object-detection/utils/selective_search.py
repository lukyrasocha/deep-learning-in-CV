from load_data import Potholes
from torch.utils.data import DataLoader
from torchvision import transforms
from visualize import visualize_samples
from tensordict import TensorDict
import argparse
import random
import time 
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches


# ref https://pyimagesearch.com/2020/06/29/opencv-selective-search-for-object-detection/

if __name__ == "__main__":
    # Set up the argument parser
    parser = argparse.ArgumentParser(description='Selective Search on Dataset Images')
    parser.add_argument('-i', '--index', type=int, default=0,
                        help='Index of the image in the dataset to process')
    parser.add_argument('-m', '--method', type=str, default='fast',
                        choices=['fast', 'quality'],
                        help='Selective search method (fast or quality)')
    args = parser.parse_args()

    # Define any transforms
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])

    # Initialize the dataset and dataloader
    potholes_dataset = Potholes(split='Train', transform=transform, folder_path='Potholes')
    dataloader = DataLoader(potholes_dataset, batch_size=32, shuffle=True, num_workers=8)

    # Get the image and its targets from the dataset using the provided index
    try:
        image_tensor, targets = potholes_dataset[args.index]
    except IndexError:
        print(f"Error: Index {args.index} is out of bounds for the dataset size {len(potholes_dataset)}.")
        exit(1)

    # Convert the tensor image to a NumPy array
    image_np = image_tensor.numpy()
    # Transpose the image to [H, W, C]
    image_np = np.transpose(image_np, (1, 2, 0))
    # Convert to uint8 if necessary
    image_np = (image_np * 255).astype(np.uint8)

    # Initialize OpenCV's selective search implementation and set the input image
    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
    # Convert the image to BGR for OpenCV processing
    image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    ss.setBaseImage(image_bgr)

    # Choose the selective search method based on the argument
    method = args.method
    if method == "fast":
        print("[INFO] using *fast* selective search")
        ss.switchToSelectiveSearchFast()
    else:
        print("[INFO] using *quality* selective search")
        ss.switchToSelectiveSearchQuality()

    # Run selective search on the input image
    start = time.time()
    rects = ss.process()
    end = time.time()
    print(f"[INFO] selective search took {end - start:.4f} seconds")
    print(f"[INFO] {len(rects)} total region proposals")
    # print what is the type of rects
    print(type(rects))
    print(rects)
    # print the first 5 elements of rects
    print(rects[:5])

    # Loop over the region proposals in chunks to visualize them
    for i in range(0, len(rects), 300):
        # Clone the original image to draw on it
        output = image_np.copy()

        # Create a figure and axis to plot on
        fig, ax = plt.subplots(1, figsize=(10, 8))
        # Display the image
        ax.imshow(output)
        # Loop over the current subset of region proposals
        for (x, y, w, h) in rects[i:i + 300]:
            # Create a Rectangle patch
            rect = patches.Rectangle((x, y), w, h, linewidth=1,
                                     edgecolor=np.random.rand(3,), facecolor='none')
            # Add the patch to the Axes
            ax.add_patch(rect)
        plt.title(f"Region Proposals {i+1} to {min(i+300, len(rects))}")
        plt.axis('off')  # Hide axes ticks
        plt.tight_layout()
        plt.savefig(f"region_proposals_{i+1}_to_{min(i+300, len(rects))}.png")
        plt.show()

        # Prompt the user to continue or quit
        user_input = input("Press Enter to continue, or 'q' to quit: ")
        if user_input.lower() == 'q':
            break