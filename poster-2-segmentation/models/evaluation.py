import torch
from utils.logger import logger
import matplotlib.pyplot as plt

# Step 1: Define a function to evaluate a single model
def evaluate_model(model, data_loader, device, metrics, dataset_name):
    model.eval()
    metric_totals = {metric.__name__: 0.0 for metric in metrics}
    num_batches = len(data_loader)

    with torch.no_grad():
        for images, masks in data_loader:
            images, masks = images.to(device), masks.to(device)
            outputs = torch.sigmoid(model(images))  # Sigmoid to get probabilities
            preds = (torch.sigmoid(outputs) > 0.5).float()

            for metric in metrics:
                metric_value = metric(preds, masks)
                metric_totals[metric.__name__] += metric_value

    # Calculate average for each metric
    metric_averages = {metric: total / num_batches for metric, total in metric_totals.items()}

    # Logging and printout
    logger.info(f"Evaluation results for {dataset_name}:")
    for metric, average in metric_averages.items():
        logger.info(f"{metric}: {average:.4f}")
    
    return metric_averages

# Step 2: Define a function to compare multiple models and plot results
def compare_models(models, model_names, data_loader, device, metrics, dataset_name):
    all_metrics = {metric.__name__: [] for metric in metrics}
    
    # Evaluate each model and collect results
    for model, name in zip(models, model_names):
        logger.working_on(f"Evaluating model {name} on {dataset_name}")
        metric_averages = evaluate_model(model, data_loader, device, metrics, dataset_name)
        
        # Append each metric result to all_metrics for plotting
        for metric, value in metric_averages.items():
            all_metrics[metric].append(value)
    
    # Plotting the comparison results
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