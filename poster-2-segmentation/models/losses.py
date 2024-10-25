import torch.nn.functional as F

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