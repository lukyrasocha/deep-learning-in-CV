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
from visualize import visualize_proposals, visualize_proposal
from tensordict import TensorDict
from metrics import IoU
import sys

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

def generate_proposals_and_targets(original_image, original_targets , transform, method = 'fast', max_proposals = 2000):
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
    original_image_np = np.array(original_image)

    # Initialize selective search
    selection_search = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
    selection_search.setBaseImage(original_image_np)

    if method == 'fast':
        selection_search.switchToSelectiveSearchFast()
    elif method == 'quality':
        selection_search.switchToSelectiveSearchQuality()

    # Run selective search to get bounding boxes
    coord_proposals = selection_search.process()

    # Limit the number of proposals
    coord_proposals = coord_proposals[:max_proposals]

    # Convert rects to proposals in TensorDict format
    proposal_images = []
    proposal_targets = []

    for (x, y, w, h) in coord_proposals:
        proposal_image = np.copy(original_image_np[y:y+h, x:x+w])

        proposal_target = {
            'xmin': torch.tensor(float(x)),
            'ymin': torch.tensor(float(y)),
            'xmax': torch.tensor(float(x + w)),
            'ymax': torch.tensor(float(y + h)),
        }

        proposal_images.append(proposal_image)
        proposal_targets.append(proposal_target)

    #visualize_proposals(original_image, original_targets, num_proposals=500, box_thickness=3, figname='target.png')
    images, targets = apply_transform_and_label_target(proposal_images, proposal_targets, original_targets, transform)
    #print(type(images))
    return images, targets

def apply_transform_and_label_target(proposal_images, proposal_targets, original_targets, transform, iou_upper_limit = 0.7, iou_lower_limit = 0.3):


    # Loop through each proposal
    images = []
    targets = []
    for proposal_image, proposal_target in zip(proposal_images, proposal_targets):

        # Track matches for each ground truth box
        iou_values = []
        for gt_box in original_targets:

            # copy an instance for the proposal target so we don't overwrite it
            proposal_target_copy = proposal_target.copy()

            gt_bbox = {
                'xmin': gt_box['xmin'].item(),
                'ymin': gt_box['ymin'].item(),
                'xmax': gt_box['xmax'].item(),
                'ymax': gt_box['ymax'].item()
            }
            iou = IoU(proposal_target_copy, gt_bbox)
            iou_values.append(iou)

        if iou_values:  # Ensure the list is not empty
            iou_max = max(iou_values)
            iou_max_index = iou_values.index(iou_max)  # Find the index of the maximum IoU

            # Add to matches if IoU is above threshold
            if iou_max >= iou_upper_limit:
                proposal_target_copy.setdefault('label', torch.tensor(1, dtype=torch.int64))

                proposal_image_transformed, proposal_target_copy = apply_transformation_on_proposal_image(proposal_image, proposal_target_copy, transform)
                proposal_target_copy = adjust_original_target_to_proposal(proposal_image_transformed, proposal_target_copy, original_targets[iou_max_index])

                #print(proposal_target_copy)
                #print(proposal_image_transformed.shape)
                #visualize_proposal(proposal_image_transformed, proposal_target_copy, box_thickness=2, figname='proposals_stop.png')
                #sys.exit(0)
                images.append(proposal_image_transformed)
                targets.append(proposal_target_copy)

            elif iou_max <= iou_lower_limit:
                proposal_target_copy.setdefault('label', torch.tensor(0, dtype=torch.int64))
                proposal_image_transformed, proposal_target_copy = apply_transformation_on_proposal_image(proposal_image, proposal_target_copy, transform)

                images.append(proposal_image_transformed)
                targets.append(proposal_target_copy)
    
    return images, targets

def apply_transformation_on_proposal_image(image, target, transform):
    original_height, original_width, _ = image.shape
    image = Image.fromarray(image)

    image_tensor = transform(image)
    _, new_height, new_width = image_tensor.shape

    x_scale = new_width / original_width
    y_scale = new_height / original_height

    if int(target['label']) == 1:
        target.setdefault('x_scale', torch.tensor(float(x_scale))) 
        target.setdefault('y_scale', torch.tensor(float(y_scale))) 



    return image_tensor, target
                
def adjust_original_target_to_proposal(image, target_proposal, gt_bbox):

    # Scaling and translating the bounding box coordinates
    gt_xmin_scaled = (gt_bbox['xmin'] - target_proposal['xmin'])  
    gt_ymin_scaled = (gt_bbox['ymin'] - target_proposal['ymin']) 
    gt_xmax_scaled = (gt_bbox['xmax'] - target_proposal['xmin']) * target_proposal['x_scale']
    gt_ymax_scaled = (gt_bbox['ymax'] - target_proposal['ymin']) * target_proposal['y_scale']

    target_proposal.setdefault('gt_bbox_xmin', torch.tensor(float(gt_xmin_scaled)))
    target_proposal.setdefault('gt_bbox_ymin', torch.tensor(float(gt_ymin_scaled)))
    target_proposal.setdefault('gt_bbox_xmax', torch.tensor(float(gt_xmax_scaled)))
    target_proposal.setdefault('gt_bbox_ymax', torch.tensor(float(gt_ymax_scaled)))
            
    return target_proposal








        
