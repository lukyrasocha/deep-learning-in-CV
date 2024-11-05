def IoU(box1, box2):
    '''
    Calculates the Intersection over Union (IoU) of two rectangles.
    (The O in MABO)
    '''
    # Set coordinates for intersection of box1 and box2
    x_left = max(box1['xmin'], box2['xmin'])
    y_top = max(box1['ymin'], box2['ymin'])
    x_right = min(box1['xmax'], box2['xmax'])
    y_bottom = min(box1['ymax'], box2['ymax'])

    # Return 0 if no intersection
    if x_right < x_left or y_bottom < y_top:
        return 0.0
    
    # Calculate areas of intersection, box1, box2 and union
    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    box1_area = (box1['xmax'] - box1['xmin']) * (box1['ymax'] - box1['ymin'])
    box2_area = (box2['xmax'] - box2['xmin']) * (box2['ymax'] - box2['ymin'])
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