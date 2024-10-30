import torch

def compute_pos_weight(dataset):
    total_pixels = 0
    total_pos = 0

    for _, mask in dataset:
        mask = mask.view(-1)
        total_pixels += mask.numel()
        total_pos += mask.sum().item()

    total_neg = total_pixels - total_pos

    # Avoid division by zero
    total_pos = total_pos + 1e-6
    total_neg = total_neg + 1e-6

    pos_weight = total_neg / total_pos

    return torch.tensor(pos_weight, dtype=torch.float32)