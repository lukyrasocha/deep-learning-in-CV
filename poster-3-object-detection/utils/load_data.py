import os
import glob
import time
import torch
import json
import xml.etree.ElementTree as ET
import random
import pickle as pk

from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tensordict import TensorDict
from torch.utils.data import default_collate


def get_xml_data(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    # Initialize lists for the target
    targets = []

    # Iterate through each object in the XML file
    for obj in root.findall('object'):
    
        #If the box is a pothole the label is 1 (True)
        if obj.find('name').text == 'pothole':
            label = 1
        else:
            label = 0

        # Extract bounding box coordinates
        bndbox = obj.find('bndbox')
        xmin = int(bndbox.find('xmin').text)
        ymin = int(bndbox.find('ymin').text)
        xmax = int(bndbox.find('xmax').text)
        ymax = int(bndbox.find('ymax').text)

        # Append bounding box and label. TensorDict is used to convert the dictorary to a tensor
        directory = TensorDict({
            'xmin'  : torch.tensor(xmin, dtype=torch.float32),
            'ymin'  : torch.tensor(ymin, dtype=torch.float32),
            'xmax'  : torch.tensor(xmax, dtype=torch.float32),
            'ymax'  : torch.tensor(ymax, dtype=torch.float32),
            'labels': torch.tensor(label, dtype=torch.int64)
        })

        targets.append(directory)
    
    return targets

def pickle_save(final_image, final_target, save_to_path_full, train, index):
    if train:
        # Create the directory in the blackhole path if it doesn't exist
        os.makedirs(save_to_path_full, exist_ok=True)
        # Save the files to the BLACKHOLE path
        with open(os.path.join(save_to_path_full, f'train_image_{index}.pkl'), 'wb') as f:
            pk.dump(final_image, f)
        with open(os.path.join(save_to_path_full, f'train_target_{index}.pkl'), 'wb') as f:
            pk.dump(final_target, f)


        
def class_balance(proposal_images, proposal_targets, seed, count):
    # Initialize lists for the proposals and targets
    class_1_proposals = []
    class_1_targets = []
    class_0_proposals = []
    class_0_targets = []
    
    random.seed(seed)
    # Loop through each proposal and target
    for image, target, in zip(proposal_images, proposal_targets):
        if int(target['label']) == 1:
            # print if label is 1 missing and write the image id
            class_1_proposals.append(image)
            class_1_targets.append(target)
        else:
            class_0_proposals.append(image)
            class_0_targets.append(target)                

    # Class balancing
    total_class_1 = len(class_1_proposals)   # 25 % of the class 0 proposals
    total_class_0_ideal = int(total_class_1 * 3)     # 75 % of the class 0 proposals 
    total_class_0 = len(class_0_proposals)

    # If the number of class 0 proposals is greater than the ideal number of class 0 proposals
    if total_class_0 > total_class_0_ideal:
        # Randomly sample the indices of the class 0 proposals to keep 
        indicies = random.sample(range(total_class_0), total_class_0_ideal)
        # Create new lists of class 0 proposals and targets with the sampled indices
        class_0_proposals_new = [class_0_proposals[i] for i in indicies]
        class_0_targets_new = [class_0_targets[i] for i in indicies]
    else:
        class_0_proposals_new = class_0_proposals
        class_0_targets_new = class_0_targets
        
    # sanity check that the ideal and the new class 0 proposals are the same
    assert len(class_0_proposals_new) == total_class_0_ideal

    # Combine the class 0 and class 1 proposals
    image_proposals = class_0_proposals_new + class_1_proposals
    image_targets = class_0_targets_new + class_1_targets

    if image_proposals and image_targets:
        # combine the proposals and targets and shuffle them
        combined = list(zip(image_proposals, image_targets))
        random.shuffle(combined)
        image_proposals, image_targets = zip(*combined)

        return image_proposals, image_targets
    else:
        print(f"No proposals and targets found for the image {image_id}")
        print(f"- Proposals generated in total: {len(proposal_images)}")
        print(f"- Class 0 proposals: {len(class_0_proposals)}")
        print(f"- Class 1 proposals: {len(class_1_proposals)}")

        return None, None

