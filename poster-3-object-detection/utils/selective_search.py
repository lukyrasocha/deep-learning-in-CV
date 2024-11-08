
import torch
import cv2
import numpy as np

from PIL import Image
from tensordict import TensorDict
from utils.metrics import IoU
from typing import Callable, Tuple, Dict, List
from torchvision import transforms

def generate_proposals_and_targets(
    original_image: 'PIL.Image.Image',
    original_targets: List[Dict[str, torch.Tensor]],
    transform: Callable[[np.ndarray], torch.Tensor],
    original_image_name: str,
    iou_upper_limit: float,
    iou_lower_limit: float,
    method: str,
    max_proposals: int,
    generate_target: bool 
) -> Tuple[List[torch.Tensor], List[Dict[str, torch.Tensor]]]:

    """
    Generates proposals using the Selective Search algorithm, applies transformations to the proposal images, and 
    labels them based on the Intersection over Union (IoU) with ground truth targets. The function returns a list of 
    transformed proposal images and their corresponding labeled targets.

    Parameters:
    -----------
    original_image : PIL.Image.Image
        The original image from which the proposals will be generated. The image is expected to be in HxWxC format.
    
    original_targets : list of dict
        A list of dictionaries, each representing a ground truth bounding box. Each dictionary contains bounding 
        box keys ('xmin', 'ymin', 'xmax', 'ymax') representing the coordinates of the ground truth box.

    transform : callable
        A transformation function that takes an image (np.array) and returns a transformed tensor. This 
        transformation is applied to each proposal image.

    original_image_name : str
        Name of the image so we know where the proposal image comes from

    iou_upper_limit : float
        The upper threshold for Intersection over Union (IoU). Proposals with an IoU greater than this value are 
        labeled as positive (1).

    iou_lower_limit : float
        The lower threshold for IoU. Proposals with an IoU smaller than this value are labeled as negative (0).

    method : str
        The type of Selective Search to use. The available options are:
        - 'fast': Faster but lower quality proposals.
        - 'quality': Higher quality but slower proposals.

    max_proposals : int
        The maximum number of proposals to generate. The function will return up to this number of proposals.
    
    generate_target:
        For validation and test we don't know the target and therefore we can not generate it
    Returns:
    --------
    tuple
        A tuple containing:
        - images : list of torch.Tensor
            A list of transformed proposal images represented as PyTorch tensors.
        
        - targets : list of dict
            A list of dictionaries where each dictionary contains:
            - 'label': Indicates whether the proposal is positive (1) or negative (0).
            - Updated bounding box coordinates (if applicable).
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
    proposal_images_tensor = []

    for (x, y, w, h) in coord_proposals:
        proposal_image = np.copy(original_image_np[y:y+h, x:x+w])

        proposal_target = {
            'xmin': torch.tensor(float(x)),
            'ymin': torch.tensor(float(y)),
            'xmax': torch.tensor(float(x + w)),
            'ymax': torch.tensor(float(y + h)),
            'original_image_name' : original_image_name,
        }

        proposal_images.append(proposal_image)
        proposal_targets.append(proposal_target)

        if generate_target is False:
            proposal_image = Image.fromarray(proposal_image)
            proposal_images_tensor.append(transform(proposal_image))
            
        
    if generate_target is False:
        return proposal_images_tensor, proposal_targets

    elif generate_target is True:
        images, targets = apply_transform_and_label_target(proposal_images, proposal_targets, original_targets, transform, iou_upper_limit, iou_lower_limit)
        return images, targets

        


def apply_transform_and_label_target(
    proposal_images: List[np.ndarray],
    proposal_targets: List[Dict[str, torch.Tensor]],
    original_targets: List[Dict[str, torch.Tensor]],
    transform: Callable[[np.ndarray], torch.Tensor],
    iou_upper_limit: float,
    iou_lower_limit: float
) -> Tuple[List[torch.Tensor], List[Dict[str, torch.Tensor]]]:
    """
    Applies a transformation to proposal images, calculates the IoU (Intersection over Union) with ground truth targets,
    and labels the proposal target based on IoU thresholds. The transformed images and labeled targets are returned.

    Parameters:
    -----------
    proposal_images : list of numpy.ndarray
        A list of input proposal images, each represented as a 3D array (height, width, channels).
    
    proposal_targets : list of dict
        A list of dictionaries (e.g., `TensorDict`) representing target attributes for each proposal image. Each dictionary
        must include bounding box keys ('xmin', 'ymin', 'xmax', 'ymax').

    original_targets : list of dict
        A list of dictionaries representing the ground truth bounding boxes. Each dictionary must include bounding box
        keys ('xmin', 'ymin', 'xmax', 'ymax').

    transform : callable
        A transformation function (e.g., a PyTorch transform) that takes an image (e.g., PIL Image) and outputs
        a transformed tensor.

    iou_upper_limit : float
        The IoU threshold above which a proposal is considered to match a ground truth box and is labeled as positive (1).

    iou_lower_limit : float
        The IoU threshold below which a proposal is considered as negative (0).

    Returns:
    --------
    tuple
        A tuple containing:
        - images : list of torch.Tensor
            The list of transformed proposal images as PyTorch tensors.
        - targets : list of dict
            The list of updated proposal target dictionaries with added keys:
            - 'label': Indicates whether the proposal is positive (1) or negative (0).
            - Updated bounding box coordinates if applicable.
    """


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
            if iou_max > iou_upper_limit:
                proposal_target_copy.setdefault('label', torch.tensor(1, dtype=torch.int64))

                proposal_image_transformed, proposal_target_copy = apply_transformation_on_proposal_image_and_target(proposal_image, proposal_target_copy, transform, original_targets[iou_max_index])
                images.append(proposal_image_transformed)
                targets.append(proposal_target_copy)

            elif iou_max < iou_lower_limit:
                proposal_target_copy.setdefault('label', torch.tensor(0, dtype=torch.int64))
                proposal_image_transformed, proposal_target_copy = apply_transformation_on_proposal_image_and_target(proposal_image, proposal_target_copy, transform, None)

                images.append(proposal_image_transformed)
                targets.append(proposal_target_copy)
    
    return images, targets



def apply_transformation_on_proposal_image_and_target(
    proposal_image: np.ndarray,
    proposal_target: Dict[str, torch.Tensor],
    transform: Callable[[np.ndarray], torch.Tensor],
    gt_bbox: Dict[str, torch.Tensor]
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:

    """
    Applies a transformation to an input proposal image and rescales its corresponding target properties, if it exist.

    This function processes a given proposal image represented as a `numpy.ndarray`, along with a proposal target
    and a ground truth bounding box (both provided as dictionary structures). It checks if the proposal target
    is labeled (i.e., has `label` set to 1), and if so, applies a scaling transformation to the image and updates
    the proposal target's bounding box coordinates accordingly.

    Parameters:
    -----------
    proposal_image : numpy.ndarray
        The input proposal image, represented as a 3D array (height, width, channels).
    
    proposal_target : dict
        A dictionary-like structure (e.g., a `TensorDict`) representing target attributes. Must include at least:
        - 'label': The label indicating if the proposal has a valid target (1 for valid, otherwise ignored).
        - 'xmin', 'ymin', 'xmax', 'ymax': Bounding box coordinates.
    
    transform : callable
        A transformation function (e.g., a PyTorch transform) that takes an image (e.g., PIL Image) and outputs
        a transformed tensor.

    gt_bbox : dict
        A dictionary-like structure representing the ground truth bounding box with keys:
        - 'xmin', 'ymin', 'xmax', 'ymax': Coordinates of the bounding box.

    Returns:
    --------
    tuple
        A tuple containing:
        - proposal_image_tensor : torch.Tensor
            The transformed image as a PyTorch tensor.
        - proposal_target : dict
            The updated proposal target dictionary with added keys:
            - 'x_scale', 'y_scale': The scaling factors for width and height.
            - 'gt_bbox_xmin_scaled', 'gt_bbox_ymin_scaled', 'gt_bbox_xmax_scaled', 'gt_bbox_ymax_scaled': Scaled coordinates of the bounding box.
    """


    original_height, original_width, _ = proposal_image.shape
    proposal_image = Image.fromarray(proposal_image)

    proposal_image_tensor = transform(proposal_image)
    _, new_height, new_width = proposal_image_tensor.shape

    x_scale = new_width / original_width
    y_scale = new_height / original_height

    if int(proposal_target['label']) == 1:
        proposal_target.setdefault('x_scale', torch.tensor(float(x_scale))) 
        proposal_target.setdefault('y_scale', torch.tensor(float(y_scale))) 
        
        # Scaling and translating the bounding box coordinates
        gt_xmin_scaled = (gt_bbox['xmin'] - proposal_target['xmin'])  
        gt_ymin_scaled = (gt_bbox['ymin'] - proposal_target['ymin']) 
        gt_xmax_scaled = (gt_bbox['xmax'] - proposal_target['xmin']) * proposal_target['x_scale']
        gt_ymax_scaled = (gt_bbox['ymax'] - proposal_target['ymin']) * proposal_target['y_scale']
    
        proposal_target.setdefault('gt_bbox_xmin_scaled', torch.tensor(float(gt_xmin_scaled)))
        proposal_target.setdefault('gt_bbox_ymin_scaled', torch.tensor(float(gt_ymin_scaled)))
        proposal_target.setdefault('gt_bbox_xmax_scaled', torch.tensor(float(gt_xmax_scaled)))
        proposal_target.setdefault('gt_bbox_ymax_scaled', torch.tensor(float(gt_ymax_scaled)))

    return proposal_image_tensor, proposal_target
                








        
