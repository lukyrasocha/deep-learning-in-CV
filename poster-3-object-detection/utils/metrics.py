import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch
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
    """
    if not tensor_dict_list:
        return []

    # Sort the predictions by confidence score in descending order
    sorted_targets = sorted(
        tensor_dict_list,
        key=lambda x: x['pre_class'],
        reverse=True
    )

    bboxes_after_nms = []

    while sorted_targets:
        # Select the target with the highest 'pre_class'
        chosen_target = sorted_targets.pop(0)
        bboxes_after_nms.append(chosen_target)

        # Filter out overlapping boxes based on IoU
        remaining_targets = []
        for target in sorted_targets:
            iou = IoU(
                chosen_target["pre_bbox_xmin"], chosen_target["pre_bbox_ymin"],
                chosen_target["pre_bbox_xmax"], chosen_target["pre_bbox_ymax"], 
                target["pre_bbox_xmin"], target["pre_bbox_ymin"],
                target["pre_bbox_xmax"], target["pre_bbox_ymax"]
            )

            if iou < iou_threshold:
                remaining_targets.append(target)

        sorted_targets = remaining_targets

    return bboxes_after_nms


def calculate_precision_recall(ground_truths, predictions, iou_threshold):
    """
    Calculate precision, recall, and mean average precision (mAP) for object detection.

    Parameters:
    -----------
    ground_truths : List[List[Dict]]
        A list of lists containing ground truth bounding box dictionaries for each image.

    predictions : List[List[Dict]]
        A list of lists containing predicted bounding box dictionaries for each image.

    iou_threshold : float
        The IoU threshold to consider a prediction as a true positive.

    Returns:
    --------
    np.ndarray
        A numpy array with columns [precision, recall].

    float
        The mean average precision (mAP) calculated without interpolation.
    """

    TP = []
    FP = []
    total_gt = []
    pre_prob = []
    
    # Get ground truth and predictions for one image

    for gt_boxes, pred_boxes in zip(ground_truths, predictions):
        # Sort the predictions so we are looking on the boxes with the highest probability
        pred_boxes = sorted(pred_boxes, key=lambda x: x['pre_class'], reverse=True)

        # Create an set, to ensure that we are not going to match multiple prediction with the same ground truth box
        gt_matched = set()

        for i, pred_box in enumerate(pred_boxes):
            
            # This below is used to keep track of the highest IoU for a prediction. We use j as the index for the ground truth 
            best_iou = 0.0
            best_gt_idx = -1
            for j, gt_box in enumerate(gt_boxes):

                #If we don't have the ground truth in our set we will calculate the IoU
                if j not in gt_matched:
                    iou = IoU(pred_box["pre_bbox_xmin"], pred_box["pre_bbox_ymin"],
                        pred_box["pre_bbox_xmax"], pred_box["pre_bbox_ymax"], 
                        gt_box["xmin"], gt_box["ymin"],
                        gt_box["xmax"], gt_box["ymax"]
                    )

                    # If it is above the best IoU, we will remember it
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = j

            # Append the predicted probability so we can use it to sort when calculation the mAP
            pre_prob.append(pred_box['pre_class'])
            # Check if the best IoU is above a certain threshold. If it is, we will append 1 to TP and 0 to FP
            if best_iou >= iou_threshold:
                TP.append(1)
                FP.append(0)
                total_gt.append(1)
            else:
                TP.append(0)
                FP.append(1)

    # Convert the list into numpy, so we can sort based on the highest probability             
    TP = np.array(TP)
    FP = np.array(FP)
    pre_prob = np.array(pre_prob)

    # Get the sorted indices based on pre_prob in descending order
    sorted_indices = np.argsort(pre_prob)[::-1] # argsort return for lowest to highest but since we want it reverse we use the slicing syntax

    # Sort TP, FP, and pre_prob based on these indices
    TP_sorted = TP[sorted_indices]
    FP_sorted = FP[sorted_indices]
    pre_prob_sorted = pre_prob[sorted_indices]

    # Calculate the precision and recall

    precision_values = []
    recall_values = []

    cumulative_TP = 0 
    cumulative_FP = 0 
    epsilon = np.finfo(float).eps
    for i in range(len(TP_sorted)):
        # Update cumulative TP and FP up to the current index
        cumulative_TP += TP_sorted[i]
        cumulative_FP += FP_sorted[i]

        # Calculate precision and recall at the current step

        precision = cumulative_TP / (cumulative_TP + cumulative_FP + epsilon)
        recall = cumulative_TP / ( len(total_gt) + epsilon)

        # Append to the lists
        precision_values.append(precision)
        recall_values.append(recall)
    
    return precision_values, recall_values

def calculate_mAP(precision_list, recall_list):
    """
    Calculate mean Average Precision (mAP) from precision and recall lists.
    """
    # Convert lists to numpy arrays
    precision = np.array(precision_list)
    recall = np.array(recall_list)

    # Check if precision and recall lists are empty
    if len(precision) == 0 or len(recall) == 0:
        return 0.0

    # Append sentinel values at the start and end
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))

    # Compute the precision envelope
    for i in range(len(mpre) - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # Calculate mAP
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    mAP = np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1])

    return mAP

# utils/metrics.py
def compute_iou(box1, box2):
    """
    Compute IoU between two boxes.
    """
    # Intersection coordinates
    x_left = max(box1[0], box2[0])
    y_top = max(box1[1], box2[1])
    x_right = min(box1[2], box2[2])
    y_bottom = min(box1[3], box2[3])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # Areas
    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    iou = intersection_area / float(box1_area + box2_area - intersection_area + 1e-6)
    return iou





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
    plt.savefig('../figures/Metrics_plots/nms.svg')
    plt.show()



    ############################################################################################################
    #                                               Calculate mAP                                              #
    ############################################################################################################

    # Example usage
    ground_truths = [[{'xmin': 10, 'ymin': 20, 'xmax': 50, 'ymax': 60}, {'xmin': 100, 'ymin': 120, 'xmax': 150, 'ymax': 160}, {'xmin': 210, 'ymin': 220, 'xmax': 250, 'ymax': 260}],
                     [{'xmin': 10, 'ymin': 20, 'xmax': 50, 'ymax': 60}, {'xmin': 100, 'ymin': 120, 'xmax': 150, 'ymax': 160}, {'xmin': 210, 'ymin': 220, 'xmax': 250, 'ymax': 260}]]

    detections = [[
        {'pre_bbox_xmin': 320, 'pre_bbox_ymin': 330, 'pre_bbox_xmax': 340, 'pre_bbox_ymax': 350, 'pre_class': 0.5}],
        [{'pre_bbox_xmin': 100, 'pre_bbox_ymin': 120, 'pre_bbox_xmax': 150, 'pre_bbox_ymax': 160, 'pre_class': 0.99},
        {'pre_bbox_xmin': 320, 'pre_bbox_ymin': 330, 'pre_bbox_xmax': 340, 'pre_bbox_ymax': 350, 'pre_class': 0.9},
        {'pre_bbox_xmin': 10,  'pre_bbox_ymin': 20,  'pre_bbox_xmax': 50,  'pre_bbox_ymax': 60,  'pre_class': 0.95},
        {'pre_bbox_xmin': 210, 'pre_bbox_ymin': 220, 'pre_bbox_xmax': 250, 'pre_bbox_ymax': 260, 'pre_class': 0.1}
    ]]

    # Calculate mAP
    precision, recall = calculate_precision_recall(ground_truths, detections, 0.5)

    plt.figure(figsize=(12,8))
    plt.plot(recall, precision, 'r-')
    plt.title('Precision vs recall')
    plt.xlabel('recall')
    plt.ylabel('precision')
    plt.savefig('../figures/Metrics_plots/precision_recall.svg')
    
    print('The precision vector is:', precision)
    print()
    print('The recall vector is:', recall)
    print('The mAP value is:', calculate_mAP(precision, recall))








        