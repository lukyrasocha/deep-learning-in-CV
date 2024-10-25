import matplotlib.pyplot as plt
import random
import torch
import numpy as np

def display_random_images_and_masks(dataset, figname, num_images=3):
    random.seed(42)
    random_indices = random.sample(range(len(dataset)), num_images)
    plt.figure(figsize=(10, num_images * 5))

    for i, idx in enumerate(random_indices):
        image, mask = dataset[idx]

        image_np = image.permute(1, 2, 0).numpy()  # Change from CxHxW to HxWxC for plotting
        mask_np = mask.squeeze().numpy()  

        # Display image
        plt.subplot(num_images, 2, 2 * i + 1)
        plt.imshow(image_np)
        plt.axis('off')
        plt.title(f"Image {idx}")

        # Display mask
        plt.subplot(num_images, 2, 2 * i + 2)
        plt.imshow(mask_np, cmap='gray')
        plt.axis('off')
        plt.title(f"Mask {idx}")

    plt.tight_layout()
    plt.savefig(f"figures/{figname}")

def display_image_mask_prediction(image, mask, prediction, figname):
    # Convert the prediction to a numpy array for processing
    pred_np = prediction.squeeze().cpu().numpy()

    plt.figure(figsize=(15, 5))

    # Image
    plt.subplot(1, 3, 1)
    plt.imshow(image.permute(1, 2, 0).cpu().numpy())
    plt.grid(True, which='both', color='gray', linestyle='--', linewidth=0.5)
    plt.xticks(fontsize=8)
    plt.yticks(fontsize=8)
    plt.title(f"Image")

    # Ground Truth Mask
    plt.subplot(1, 3, 2)
    plt.imshow(mask.squeeze().cpu().numpy(), cmap='gray')
    plt.grid(True, which='both', color='gray', linestyle='--', linewidth=0.5)
    plt.xticks(fontsize=8)
    plt.yticks(fontsize=8)
    plt.title("Ground Truth Mask")

    # Predicted Segmentation
    plt.subplot(1, 3, 3)
    plt.imshow(pred_np, cmap='gray')
    plt.grid(True, which='both', color='gray', linestyle='--', linewidth=0.5)
    plt.xticks(fontsize=8)
    plt.yticks(fontsize=8)
    plt.title("Stitched Segmentation")

    plt.tight_layout()
    plt.savefig(f"figures/{figname}")


def visualize_predictions(model, data_loader, device, figname, num_images=3):
    model.eval()  
    images_shown = 0

    plt.figure(figsize=(10, num_images * 5))

    with torch.no_grad():  
        for images, masks in data_loader:
            images = images.to(device)
            masks = masks.to(device)

            preds = torch.sigmoid(model(images))
            preds = (preds > 0.5).float()  # Threshold predictions to 0 or 1

            for i in range(images.size(0)):
                if images_shown >= num_images:
                    break
                
                # Convert tensor to numpy array for plotting
                image_np = images[i].permute(1, 2, 0).cpu().numpy()  # Change from CxHxW to HxWxC
                mask_np = masks[i].squeeze().cpu().numpy() 
                pred_np = preds[i].squeeze().cpu().numpy()

                _, pred_width, pred_height = preds[i].shape
                _, original_width, original_height = masks[i].shape

                # Plot original image
                plt.subplot(num_images, 3, images_shown * 3 + 1)
                plt.imshow(image_np)
                plt.axis('off')
                plt.title("Image")

                # Plot ground truth mask
                plt.subplot(num_images, 3, images_shown * 3 + 2)
                plt.imshow(mask_np, cmap='gray')
                plt.axis('off')
                plt.title("True Mask")

                # Plot predicted mask
                plt.subplot(num_images, 3, images_shown * 3 + 3)
                pred_np = np.pad(pred_np, pad_width = int((original_width-pred_width)/2), mode="constant", constant_values = 0)
                plt.imshow(pred_np, cmap='gray')
                plt.axis('off')
                plt.title("Predicted Mask")

                images_shown += 1
            
            if images_shown >= num_images:
                break

    plt.tight_layout()
    plt.savefig(f"figures/{figname}")