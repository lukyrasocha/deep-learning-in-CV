import matplotlib.pyplot as plt
import random
import torch

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
def display_image_and_mask(image, mask, figname, image_id, num_images=1):
    plt.figure(figsize=(10, num_images * 5))

    image_np = image.permute(1, 2, 0).numpy()  # Change from CxHxW to HxWxC for plotting
    mask_np = mask.squeeze().numpy()  

    # Display image
    plt.subplot(num_images, 2, 1)
    plt.imshow(image_np)
    plt.axis('off')
    plt.title(f"Image ID: {image_id}")  # Use the image ID in the title

    # Display mask
    plt.subplot(num_images, 2, 2)
    plt.imshow(mask_np, cmap='gray')
    plt.axis('off')
    plt.title(f"Mask")

    plt.tight_layout()
    plt.savefig(f"figures/{figname}")
    plt.close()

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
                plt.imshow(pred_np, cmap='gray')
                plt.axis('off')
                plt.title("Predicted Mask")

                images_shown += 1
            
            if images_shown >= num_images:
                break

    plt.tight_layout()
    plt.savefig(f"figures/{figname}")