import torch
import matplotlib.pyplot as plt
import wandb

from utils.logger import logger
from models.split_image import split_image_into_patches  

def evaluate_model(model, data_loader, device, metrics, dataset_name, patch_size, add_edge = False):
    model.eval()
    metric_totals = {metric.__name__: 0.0 for metric in metrics}
    num_images = 0

    with torch.no_grad():
        for images, masks in data_loader:
            images = images.to(device)
            masks = masks.to(device)

            # Since batch_size=1, remove the batch dimension
            image = images.squeeze(0)
            mask = masks.squeeze(0)

            # Process the image by splitting into patches
            predicted_mask = split_image_into_patches(image, patch_size, model, add_edge=add_edge)

            assert mask.shape == predicted_mask.shape, "Predicted mask needs to have same shape as the target mask"

            # Compute metrics
            for metric in metrics:
                metric_value = metric(predicted_mask.unsqueeze(0), mask.unsqueeze(0))
                metric_totals[metric.__name__] += metric_value

            num_images += 1

    # Calculate average metrics
    metric_averages = {metric: total / num_images for metric, total in metric_totals.items()}

    wandb.log({f"{dataset_name}/{metric}": average for metric, average in metric_averages.items()})

    logger.success(f"Evaluation results for {dataset_name}:")
    for metric, average in metric_averages.items():
        logger.info(f"{metric}: {average:.4f}")

    return metric_averages

def compare_models(models, model_names, data_loader, device, metrics, dataset_name, patch_size):
    all_metrics = {metric.__name__: [] for metric in metrics}
    
    for model, name in zip(models, model_names):
        logger.working_on(f"Evaluating model {name} on {dataset_name}")
        metric_averages = evaluate_model(model, data_loader, device, metrics, dataset_name, patch_size)
        
        for metric, value in metric_averages.items():
            all_metrics[metric].append(value)
    
    plt.figure(figsize=(10, 6))
    for metric, values in all_metrics.items():
        plt.plot(model_names, values, marker='o', label=metric)
    
    plt.title(f"Model Comparison on {dataset_name}")
    plt.xlabel("Model")
    plt.ylabel("Metric Value")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"figures/{dataset_name}_model_comparison.png")
    plt.show()
    logger.success(f"Saved comparison plot for {dataset_name} to 'figures/{dataset_name}_model_comparison.png'")