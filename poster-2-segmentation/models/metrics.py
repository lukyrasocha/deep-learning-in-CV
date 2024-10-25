def reshape_input(y_pred, y_real):

        _, _, y_pred_height, y_pred_width = y_pred.shape
        _, _, y_real_height, y_real_width = y_real.shape

        diff_height = (y_real_height - y_pred_height) // 2
        diff_width = (y_real_width - y_pred_width) // 2

        y_real = y_real[:, :, diff_height:y_real_height-diff_height, diff_width:y_real_width-diff_width]

        return y_real


def dice_overlap(y_pred, y_real):
    print(f'The shape of y_pred {y_pred.shape}')
    print(f'The shape of y_real {y_real.shape}')
    
    if y_pred.shape != y_real.shape:
        print(f'The shape of y_pred {y_pred.shape}')
        y_real = reshape_input(y_pred, y_real)
        print(f'The shape of y_pred {y_pred.shape}')

    pred = y_pred.contiguous().view(-1)
    target = y_real.contiguous().view(-1)
    
    intersection = (pred * target).sum()

    dice = (2. * intersection) / (pred.sum() + target.sum())
    
    return dice.item()  

# Intersection over Union (IoU)
def IoU(y_pred, y_real, epsilon=1e-6):

    if y_pred.shape != y_real.shape:
        y_real = reshape_input(y_pred, y_real)

    pred = y_pred.contiguous().view(-1)
    target = y_real.contiguous().view(-1)
    
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection

    iou = (intersection + epsilon) / (union + epsilon)
    
    return iou.item() 

def accuracy(y_pred, y_real):
    if y_pred.shape != y_real.shape:
        y_real = reshape_input(y_pred, y_real)

    pred = y_pred.contiguous().view(-1)
    target = y_real.contiguous().view(-1)

    TP_TN = ((pred == target)).sum().float()  # True Positives and True Negatives
    accuracy = TP_TN / target.numel()  # Total number of pixels

    return accuracy.item()

def sensitivity(y_pred, y_real, epsilon=1e-6):

    if y_pred.shape != y_real.shape:
        y_real = reshape_input(y_pred, y_real)

    pred = y_pred.contiguous().view(-1)
    target = y_real.contiguous().view(-1)

    TP = ((pred == 1) & (target == 1)).sum().float()
    FN = ((pred == 0) & (target == 1)).sum().float()

    sensitivity = TP / (TP + FN + epsilon)
    
    return sensitivity.item()  

def specificity(y_pred, y_real, epsilon=1e-6):

    if y_pred.shape != y_real.shape:
        y_real = reshape_input(y_pred, y_real)
    
    pred = y_pred.contiguous().view(-1)
    target = y_real.contiguous().view(-1)

    TN = ((pred == 0) & (target == 0)).sum().float()
    FP = ((pred == 1) & (target == 0)).sum().float()

    specificity = TN / (TN + FP + epsilon)
    
    return specificity.item()  