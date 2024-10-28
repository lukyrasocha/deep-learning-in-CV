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
        #plt.imshow(image_np)
        plt.imshow((image_np * 255).astype(np.uint8))
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
    #plt.imshow(image.permute(1, 2, 0).cpu().numpy())
    plt.imshow((image.permute(1, 2, 0).cpu().numpy()) * 255).astype(np.uint8)
    #plt.imshow((image_np * 255).astype(np.uint8))
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
                #plt.imshow(image_np)
                plt.imshow((image_np * 255).astype(np.uint8))
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

def visualize_weak_supervision_predictions(model, 
    data_loader, 
    device,
    CLICKS,
    figname="weak_supervision_predictions.png", 
    num_images=5, SAMPLIG='random'):
    model.eval()
    model.to(device)
    images_shown = 0

    plt.figure(figsize=(20, num_images * 5))  # Adjusted figsize for four columns
    plt.suptitle(f"Weak Supervision Predictions with {CLICKS} Clicks", fontsize=42, y=0.99)  # Adjusted title

    with torch.no_grad():
        for batch in data_loader:
            #print("BATCH********")
            #print(batch)
            #print("BATCH********")
            images, masks = batch  # Adjusted to two elements
            weak_supervision_masks = masks
            #print(masks)
            if images_shown >= num_images:
                break

            images = images.to(device)
            outputs = model(images)
            preds = torch.sigmoid(outputs)
            preds = (preds > 0.5).float()
            #print('Unique values in predictions:', torch.unique(preds))

            for i in range(images.size(0)):
                if images_shown >= num_images:
                    break

                image_np = images[i].cpu().permute(1, 2, 0).numpy()
                weak_supervision_mask_np = weak_supervision_masks[i].cpu().squeeze().numpy()
                pred_np = preds[i].cpu().squeeze().numpy()

                # Handle NaNs in weak supervision mask
                masked_ws_mask = np.ma.masked_invalid(weak_supervision_mask_np)
                cmap_ws = plt.cm.gray.copy()
                cmap_ws.set_bad(color='gray')

                # Plotting
                idx = images_shown
                # Original Image
                plt.subplot(num_images, 3, idx * 3 + 1)
                #plt.imshow(image_np)
                plt.imshow((image_np * 255).astype(np.uint8))
                plt.axis('off')
                plt.title("Original Image")

                # Weak Supervision Mask
                plt.subplot(num_images, 3, idx * 3 + 2)
                plt.imshow(masked_ws_mask, cmap=cmap_ws, vmin=0, vmax=1)
                plt.axis('off')
                plt.title("Weak Supervision Mask")

                # Model Prediction
                plt.subplot(num_images, 3, idx * 3 + 3)
                plt.imshow(pred_np, cmap='gray', vmin=0, vmax=1)
                plt.axis('off')
                plt.title("Model Prediction")

                images_shown += 1

    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust layout to make space for suptitle
    plt.savefig(f"figures/{figname}_{CLICKS}_clicks and {SAMPLIG} sampling.png")
    plt.show()

def display_random_images_and_weak_supervision_masks(dataset, figname, num_images=3):
    random.seed(42)
    random_indices = random.sample(range(len(dataset)), num_images)
    plt.figure(figsize=(15, num_images * 5))

    for i, idx in enumerate(random_indices):
        # Retrieve image and original mask
        image, mask = dataset[idx]
        # print shapes 
        print(f"Image shape: {image.shape}")
        print(f"Mask shape: {mask.shape}")
        # Print unique values in mask 
        print(f"Unique values in mask {idx}: {np.unique(mask.numpy())}")

        # Convert image and mask for plotting
        image_np = image.permute(1, 2, 0).cpu().numpy()  # CxHxW to HxWxC
        mask_np = mask.squeeze(0).cpu().numpy()  # [1, H, W] to [H, W]
        # print shapes
        print(f"Image shape: {image_np.shape}")
        print(f"Mask shape: {mask_np.shape}")

        
        # Convert to NumPy and mask NaN values
        weak_supervision_mask_np = mask_np  # Already in [H, W]
        masked_image = np.ma.masked_where(np.isnan(weak_supervision_mask_np), weak_supervision_mask_np)
        # print shapes
        print(f"Weak Supervision Mask shape: {weak_supervision_mask_np.shape}")
        print(f"Masked Image shape: {masked_image.shape}")


        
        # Configure the colormap for NaN values
        cmap = plt.cm.gray
        cmap.set_bad(color='gray', alpha=0.5)  # NaNs in transparent gray

        # Plot the original image
        plt.subplot(num_images, 2, 2 * i + 1)
        #plt.imshow(image_np)
        plt.imshow((image_np * 255).astype(np.uint8))
        plt.axis('off')
        plt.title(f"Image {idx}")

        # Plot the weak supervision mask with NaNs in transparent gray
        plt.subplot(num_images, 2, 2 * i + 2)
        plt.imshow(masked_image, cmap=cmap, interpolation='none')
        plt.axis('off')
        plt.title(f"Weak Supervision Mask {idx}")

    plt.tight_layout()
    plt.savefig(f"figures/{figname}", dpi=300)
    plt.show()