def IoU(box1, box2):
    '''
    Calculates the Intersection over Union (IoU) of two rectangles.
    '''
    # Set coordinates for intersection of box1 and box2
    x_left = max(box1['xmin'], box2['xmin'])
    y_top = max(box1['ymin'], box2['ymin'])
    x_right = min(box1['xmax'], box2['xmax'])
    y_bottom = min(box1['ymax'], box2['ymax'])

    # Return 0 if no intersection
    if x_right < x_left or y_bottom < y_top:
            return 0.0
    
    # Calculate ares of intersection, box1, box2 and union
    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    box1_area = (box1['xmax'] - box1['xmin']) * (box1['ymax'] - box1['ymin'])
    box2_area = (box2['xmax'] - box2['xmin']) * (box2['ymax'] - box2['ymin'])
    union_area = box1_area + box2_area - intersection_area

    # Intersection over union (IoU)
    iou = intersection_area / union_area

    return iou

def get_best_IoU(proposal, ground_truth):
     '''
     Gets the highest IoU of a proposal with the ground truth.
     (In case there are multiple ground truth boxes and the proposal overlaps with more than one)
     '''
     best_iou = max(IoU(proposal, gt_box) for gt_box in ground_truth)

     return best_iou
