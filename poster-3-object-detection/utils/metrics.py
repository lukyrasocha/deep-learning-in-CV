def IoU(
    box1_xmin: float,
    box1_ymin: float,
    box1_xmax: float,
    box1_ymax: float,
    box2_xmin: float,
    box2_ymin: float,
    box2_xmax: float,
    box2_ymax: float
) -> float:
    """
    Calculates the Intersection over Union (IoU) of two rectangles.

    This function computes the IoU between two bounding boxes, a standard metric in object detection to
    quantify the overlap between a predicted box and a ground truth box. It is defined as the ratio of the
    intersection area to the union area of the two boxes.

    Parameters:
    -----------
    box1_xmin : float
        The x-coordinate of the top-left corner of the first bounding box.
    
    box1_ymin : float
        The y-coordinate of the top-left corner of the first bounding box.
    
    box1_xmax : float
        The x-coordinate of the bottom-right corner of the first bounding box.
    
    box1_ymax : float
        The y-coordinate of the bottom-right corner of the first bounding box.
    
    box2_xmin : float
        The x-coordinate of the top-left corner of the second bounding box.
    
    box2_ymin : float
        The y-coordinate of the top-left corner of the second bounding box.
    
    box2_xmax : float
        The x-coordinate of the bottom-right corner of the second bounding box.
    
    box2_ymax : float
        The y-coordinate of the bottom-right corner of the second bounding box.

    Returns:
    --------
    float
        The Intersection over Union (IoU) value, ranging between 0.0 (no overlap) and 1.0 (perfect overlap).
        Returns 0.0 if the bounding boxes do not intersect.

    Explanation:
    ------------    
        IoU = intersection_area / union_area

      where:
        union_area = box1_area + box2_area - intersection_area
    """

    # Set coordinates for intersection of box1 and box2
    x_left = max(box1_xmin, box2_xmin)
    y_top = max(box1_ymin, box2_ymin)
    x_right = min(box1_xmax, box2_xmax)
    y_bottom = min(box1_ymax, box2_ymax)

    # Return 0 if no intersection
    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # Calculate areas of intersection, box1, box2, and union
    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    box1_area = (box1_xmax - box1_xmin) * (box1_ymax - box1_ymin)
    box2_area = (box2_xmax - box2_xmin) * (box2_ymax - box2_ymin)
    union_area = box1_area + box2_area - intersection_area

    # Intersection over Union (IoU)
    iou = intersection_area / union_area
    return iou


def best_proposal(proposals, ground_truth_box, return_box=False):
    '''
    Gets the highest IoU of one ground truth with all proposals.
    Can also return the box coordinates for the best proposal.
    (The BO in MABO)
    '''
    if not proposals:
        return (0.0, None) if return_box else 0.0
    
    best_iou = 0.0
    best_box = None

    for prop in proposals:
        iou = IoU(ground_truth_box, prop)
        if iou > best_iou:
            best_iou = iou
            best_box = prop
    
    if return_box:
        return best_iou, best_box
    else:
        return best_iou


def abo(ground_truth_boxes, proposals):
    '''
    Gets the average best IoU over all objects in an image.
    '''
    if not ground_truth_boxes:
        return 0

    # Calculate the sum of best IoUs for each ground truth box
    abo = sum(best_proposal(proposals, gt_box) for gt_box in ground_truth_boxes) / len(ground_truth_boxes)

    return abo

def mabo():
    '''
    Returns the mean abo for all classes in an image.
    '''
    return

def recall(ground_truth_boxes, proposals, k=0.5):
    '''
    Returns the recall percentage for all objects of one class.
    '''
    if not ground_truth_boxes:
        return 0

    # Count ground truth boxes with at least one "good" proposal
    num_good = sum(1 for gt_box in ground_truth_boxes if best_proposal(proposals, gt_box) > k)
    recall = num_good / len(ground_truth_boxes)
    return recall