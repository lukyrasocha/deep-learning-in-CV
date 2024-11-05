import pickle
import matplotlib as mpl
from load_data import Potholes
from torchvision import transforms
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from torchvision.transforms.functional import to_pil_image
from PIL import Image
from visualize import visualize_proposals
from tensordict import TensorDict
from metrics import IoU

################################################################
### move later to visualize.py

mpl.rcParams.update(mpl.rcParamsDefault)
plt.rcParams.update({'text.usetex': True})

# Set global font size for title, x-label, and y-label
plt.rcParams['axes.titlesize'] = 20
plt.rcParams['axes.labelsize'] = 16

# Set global font size for x and y tick labels
plt.rcParams['xtick.labelsize'] = 16
plt.rcParams['ytick.labelsize'] = 16

# Set global font size for the legend
plt.rcParams['legend.fontsize'] = 11

# Set global font size for the figure title
plt.rcParams['figure.titlesize'] = 42

color_primary = '#990000'  # University red


################################################################

def get_proposals_and_targets(image, target, transform, method = 'fast', max_proposals = 2000):
    """
    Generates proposals using the Selective Search algorithm.

    Args:
        image (image): The image for which to generate proposals.
        max_proposals (int): The maximum number of proposals to return.
        type (str): The type of Selective Search to use. Options are 'fast' and 'quality'.

    Returns:
        list: A list of TensorDict objects, each containing bounding box coordinates.
    """
    # Convert tensor to a NumPy array (HxWxC format)
    image_np = np.array(image)

    # Initialize selective search
    selection_search = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
    selection_search.setBaseImage(image_np)

    if method == 'fast':
        selection_search.switchToSelectiveSearchFast()
    elif method == 'quality':
        selection_search.switchToSelectiveSearchQuality()

    # Run selective search to get bounding boxes
    rects = selection_search.process()

    # Limit the number of proposals
    rects = rects[:max_proposals]

    # Convert rects to proposals in TensorDict format
    target_proposals = []
    image_proposals = []

    for (x, y, w, h) in rects:
        image_proposal = image_np[y:y+h, x:x+w]

        target_proposal = {
            'xmin': torch.tensor(float(x)),
            'ymin': torch.tensor(float(y)),
            'xmax': torch.tensor(float(x + w)),
            'ymax': torch.tensor(float(y + h)),
        }

        image_proposals.append(image_proposal)
        target_proposals.append(target_proposal)

    image_proposals, target_proposals = get_image_and_target_label(image_proposals, target_proposals, target, transform)
    
    return None, None

def get_image_and_target_label(image_proposals, target_proposals, target, transform, iou_upper_limit = 0.7, iou_lower_limit = 0.3):


    # Loop through each proposal
    images = []
    targets = []
    for image_proposal, target_proposal in zip(image_proposals, target_proposals):

        # Track matches for each ground truth box
        iou_values = []
        for gt_box in target:
            current_target = target_proposal.copy()

            gt_bbox = {
                'xmin': gt_box['xmin'].item(),
                'ymin': gt_box['ymin'].item(),
                'xmax': gt_box['xmax'].item(),
                'ymax': gt_box['ymax'].item()
            }
            iou = IoU(target_proposal, gt_bbox)
            iou_values.append(iou)

        if iou_values:  # Ensure the list is not empty
            iou_max = max(iou_values)
            iou_max_index = iou_values.index(iou_max)  # Find the index of the maximum IoU

            # Add to matches if IoU is above threshold
            if iou_max >= iou_upper_limit:
                current_target.setdefault('label', torch.tensor(1, dtype=torch.int64))

                image_transformed, current_target = apply_transformation(image_proposal, current_target, transform)
                current_target = check_if_gt_bbx_outside_proposal(current_target, target[iou_max_index])

            elif iou_max <= iou_lower_limit:
                current_target.setdefault('label', torch.tensor(0, dtype=torch.int64))
                image_transformed, current_target = apply_transformation(image_proposal, current_target, transform)

            targets.append(current_target)
    
    return images, targets


                
def check_if_gt_bbx_outside_proposal(target_proposal, gt_bbox):

    gt_xmin_scaled = gt_bbox['xmin'] * target_proposal['x_scale'] - target_proposal['xmin']
    gt_ymin_scaled = gt_bbox['ymin'] * target_proposal['y_scale'] - target_proposal['ymin']
    gt_xmax_scaled = gt_bbox['xmax'] * target_proposal['x_scale'] - target_proposal['xmin']
    gt_ymax_scaled = gt_bbox['ymax'] * target_proposal['y_scale'] - target_proposal['ymin']

    if gt_xmin_scaled < 0:
        target_proposal.setdefault('gt_bbox_xmin', torch.tensor(float(0)))            
    else:
        target_proposal.setdefault('gt_bbox_xmin', torch.tensor(float(gt_xmin_scaled)))

    if gt_ymin_scaled < 0:
        target_proposal.setdefault('gt_bbox_ymin', torch.tensor(float(0)))            
    else:
        target_proposal.setdefault('gt_bbox_ymin', torch.tensor(float(gt_ymin_scaled)))

    diff_x = abs(target_proposal['xmax'] - target_proposal['xmin'])
    if gt_xmax_scaled > diff_x:
        target_proposal.setdefault('gt_bbox_xmax', torch.tensor(float(diff_x)))            
    else:
        target_proposal.setdefault('gt_bbox_xmax', torch.tensor(float(gt_xmax_scaled)))

    diff_y = abs(target_proposal['ymax'] - target_proposal['ymin'])
    if gt_ymax_scaled > diff_y:
        target_proposal.setdefault('gt_bbox_ymax', torch.tensor(float(diff_y)))            
    else:
        target_proposal.setdefault('gt_bbox_ymax', torch.tensor(float(gt_ymax_scaled)))
            
    return target_proposal

def apply_transformation(image, target, transform):
    original_height, original_width, _ = image.shape
    image = Image.fromarray(image)
    image_tensor = transform(image)
    _, new_hight, new_width = image_tensor.shape

    x_scale = new_width / original_width
    y_scale = new_hight / original_height

    if int(target['label']) != 0:
        target.setdefault('x_scale', torch.tensor(float(x_scale))) 
        target.setdefault('y_scale', torch.tensor(float(y_scale))) 

    return image_tensor, target






        
