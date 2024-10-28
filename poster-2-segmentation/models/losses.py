import torch.nn.functional as F
from torchvision.ops import sigmoid_focal_loss
import torch

def reshape_input(y_pred, y_real):
        _, _, y_pred_height, y_pred_width = y_pred.shape
        _, _, y_real_height, y_real_width = y_real.shape

        diff_height = (y_real_height - y_pred_height) // 2
        diff_width = (y_real_width - y_pred_width) // 2

        y_real = y_real[:, :, diff_height:y_real_height-diff_height, diff_width:y_real_width-diff_width]

        return y_real

def bce_loss(y_pred, y_real):
    # Crop to the center (if they are the same size, y_real will not change because Mads said so)
    y_real = reshape_input(y_pred, y_real)

    return F.binary_cross_entropy_with_logits(y_pred, y_real)

def masked_bce_loss(inputs, targets):
    inputs_flat = inputs.view(-1)
    targets_flat = targets.view(-1)
    valid_mask = ~torch.isnan(targets_flat)
    inputs_valid = inputs_flat[valid_mask]
    targets_valid = targets_flat[valid_mask]

    #print(f'Number of valid pixels in loss function: {inputs_valid.numel()}')

    if inputs_valid.numel() == 0:
        return torch.tensor(0.0, requires_grad=True).to(inputs.device)

    loss = F.binary_cross_entropy_with_logits(inputs_valid, targets_valid)
    return loss

def focal_loss(y_pred, y_real, alpha=0.25, gamma=2):
    y_real = reshape_input(y_pred, y_real)
    # Ensure y_real is of type float32
    y_real = y_real.float()
    return sigmoid_focal_loss(y_pred, y_real, alpha=alpha, gamma=gamma, reduction='mean')

def weighted_bce_loss(y_pred, y_real, pos_weight):
    y_real = reshape_input(y_pred, y_real)
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    loss = loss_fn(y_pred, y_real)
    return loss