import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt


def smooth_grad(model, image, label, device, stdev_spread=0.15, n_samples=25, magnitude=True):
    model.eval()
    image = image.to(device).unsqueeze(0)  # Add batch dimension
    image.requires_grad = True

    stdev = stdev_spread * (image.max() - image.min()).item()
    total_gradients = torch.zeros_like(image)  # an array

    for i in range(n_samples):
        # Generate noise
        noise = torch.normal(0, 0.15, size=image.size()).to(device)
        noisy_image = image + noise

        # Forward pass
        output = model(noisy_image)

        # Adjust the label shape
        adjusted_label = label.to(device).float().unsqueeze(0).unsqueeze(1)

        # Compute loss
        loss = F.binary_cross_entropy_with_logits(output, adjusted_label)

        # Backward pass
        model.zero_grad()
        loss.backward()

        # Get gradients
        gradients = image.grad.clone()
        if magnitude:
            gradients = gradients.abs()

        total_gradients += gradients
        image.grad.zero_()

    # Average the gradients
    avg_gradients = total_gradients / n_samples
    return avg_gradients.squeeze().cpu().detach().numpy()


def visualize_saliency_map(image, saliency_map):
    # Convert tensors to numpy arrays
    image = image.permute(1, 2, 0).numpy()  # Shape: (H, W, 3)
    image = (image - image.min()) / (image.max() - image.min())  # Normalize for display

    # Compute the L2 norm across color channels
    saliency_map = np.sqrt(np.sum(saliency_map ** 2, axis=0))  # Shape: (H, W)

    # Normalize saliency map
    saliency_map = (saliency_map - saliency_map.min()) / (saliency_map.max() - saliency_map.min())

    # Plot
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.title('Original Image')
    plt.imshow(image)
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title('SmoothGrad Saliency Map')
    plt.imshow(image)
    plt.imshow(saliency_map, cmap='jet', alpha=0.5)
    plt.axis('off')

    plt.show()
