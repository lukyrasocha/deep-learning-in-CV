import torch
import torch.nn.functional as F

# Cross entropy
def bce_loss(y_pred, y_real):
    return F.binary_cross_entropy_with_logits(y_pred, y_real)

# Focal loss    ### check if correct
def focal_loss(y_pred, y_real, alpha=1, gamma=2):
    pt = torch.exp(-bce_loss(y_pred, y_real))
    return alpha * (1 - pt) ** gamma * bce_loss(y_pred, y_real)

# Cross entropy with class weight adjustments
def bce_cwa_loss(y_pred, y_real):
    return