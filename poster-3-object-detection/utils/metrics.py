from typing import Dict, List

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


def non_max_suppression(
    tensor_dict_list: List[Dict], 
    iou_threshold: float = 0.3
) -> List[Dict]:
    """
    Applies Non-Maximum Suppression (NMS) to filter overlapping bounding boxes based on their Intersection over Union (IoU) scores.

    Parameters:
    -----------
    tensor_dict_list : list of dict
        A list of dictionaries where each dictionary contains information about a bounding box and its associated confidence score.
        Each dictionary must have the following keys:
        - 'pre_class': The confidence score of the bounding box (float).
        - 'pre_bbox_xmin', 'pre_bbox_ymin', 'pre_bbox_xmax', 'pre_bbox_ymax': Coordinates of the bounding box 
          (xmin, ymin, xmax, ymax).
    
    iou_threshold : float, optional, default=0.3
        The IoU threshold for filtering overlapping bounding boxes. A box will be removed if its IoU with a higher 
        confidence box is greater than this threshold.

    Returns:
    --------
    list of dict
        A list of dictionaries, each containing the remaining bounding boxes after applying Non-Maximum Suppression.
        These boxes are sorted by their confidence scores in descending order.
    """

    sorted_targets = sorted(
        [tensor for tensor in tensor_dict_list if tensor['pre_class'] >= 0.5],  # This line removes the predictions that are below 0.5 since it will be a background
        key=lambda x: x['pre_class'],                                           # TUse the pre_class to decide the order for the list
        reverse=True                                                            # Get the highest first
    )

    bboxes_after_nms = []

    while sorted_targets:
        # Select the target with the highest 'pre_class'
        chosen_target = sorted_targets.pop(0)
        bboxes_after_nms.append(chosen_target)

        # Filter out overlapping boxes based on IoU
        remaining_targets = []
        for target in sorted_targets:
            iou = IoU(chosen_target["pre_bbox_xmin"], chosen_target["pre_bbox_ymin"],
                      chosen_target["pre_bbox_xmax"], chosen_target["pre_bbox_ymax"], 
                      target["pre_bbox_xmin"], target["pre_bbox_ymin"],
                      target["pre_bbox_xmax"], target["pre_bbox_ymax"]
                      )

            if iou < iou_threshold:
                remaining_targets.append(target)

        sorted_targets = remaining_targets

    return bboxes_after_nms


if __name__ == '__main__':

    
    example_tensor_dicts = [
        # Group 1
        {'pre_bbox_xmin': 10, 'pre_bbox_ymin': 20, 'pre_bbox_xmax': 50, 'pre_bbox_ymax': 60, 'pre_class': 0.85},
        {'pre_bbox_xmin': 12, 'pre_bbox_ymin': 22, 'pre_bbox_xmax': 52, 'pre_bbox_ymax': 62, 'pre_class': 0.9},
        {'pre_bbox_xmin': 15, 'pre_bbox_ymin': 25, 'pre_bbox_xmax': 55, 'pre_bbox_ymax': 65, 'pre_class': 0.75},

        # Group 2
        {'pre_bbox_xmin': 100, 'pre_bbox_ymin': 120, 'pre_bbox_xmax': 140, 'pre_bbox_ymax': 160, 'pre_class': 0.8},
        {'pre_bbox_xmin': 105, 'pre_bbox_ymin': 125, 'pre_bbox_xmax': 145, 'pre_bbox_ymax': 165, 'pre_class': 0.88},
        {'pre_bbox_xmin': 110, 'pre_bbox_ymin': 130, 'pre_bbox_xmax': 150, 'pre_bbox_ymax': 170, 'pre_class': 0.9}
    ]

    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

    # Run the function
    results = non_max_suppression(example_tensor_dicts)

    # Plot the results
    fig, ax = plt.subplots(1, figsize=(12, 8))
    colors = ['r', 'g', 'b', 'm', 'c', 'y']

    for i, bbox in enumerate(example_tensor_dicts):
        rect = patches.Rectangle(
            (bbox['pre_bbox_xmin'], bbox['pre_bbox_ymin']),
            bbox['pre_bbox_xmax'] - bbox['pre_bbox_xmin'],
            bbox['pre_bbox_ymax'] - bbox['pre_bbox_ymin'],
            linewidth=1,
            edgecolor=colors[i % len(colors)],
            facecolor='none',
            linestyle='--',
            label=f'Original Box {i+1}'
        )
        ax.add_patch(rect)

    for i, bbox in enumerate(results):
        rect = patches.Rectangle(
            (bbox['pre_bbox_xmin'], bbox['pre_bbox_ymin']),
            bbox['pre_bbox_xmax'] - bbox['pre_bbox_xmin'],
            bbox['pre_bbox_ymax'] - bbox['pre_bbox_ymin'],
            linewidth=2,
            edgecolor='black',
            facecolor='none',
            label=f'NMS Box {i+1}'
        )
        ax.add_patch(rect)
    plt.xlim([0, 200])
    plt.ylim([0, 200])
    plt.title('Non-Max Suppression Results')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.legend()
    plt.grid(True)
    plt.savefig('/zhome/20/1/209339/02516-intro-to-dl-in-cv/poster-3-object-detection/figures/nms_result.png')
    plt.show()












        