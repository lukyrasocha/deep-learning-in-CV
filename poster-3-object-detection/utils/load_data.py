import os
import glob
import time
import torch
import json
import xml.etree.ElementTree as ET
import random
import pickle as pk
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tensordict import TensorDict
from utils.selective_search import generate_proposals_for_test_and_val

def collate_fn(batch):
    images = [img for proposal_images in batch for img in proposal_images[0]]
    targets = [tgt for proposal_targets in batch for tgt in proposal_targets[1]]
    indices = [idx for _, _, idx in batch]

    return torch.stack(images), targets, indices

class Trainingset(Dataset):
    def __init__(self, image_dir, target_dir, transform=None):
        self.image_files = sorted(glob.glob(os.path.join(image_dir, 'train_image_*.pkl')))
        self.target_files = sorted(glob.glob(os.path.join(target_dir, 'train_target_*.pkl')))
        self.transform = transform

        assert len(self.image_files) == len(self.target_files), "Number of images and targets must be the same."

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # Load the image proposals from the pickle file
        with open(self.image_files[idx], 'rb') as img_f:
            proposal_images = pk.load(img_f)

        # Load the corresponding target proposals from the pickle file
        with open(self.target_files[idx], 'rb') as tgt_f:
            proposal_targets = pk.load(tgt_f)

        # Apply the transformation, if any, to each proposal image
        if self.transform:
            proposal_images = [self.transform(img) if isinstance(img, Image.Image) else img for img in proposal_images]

        return proposal_images, proposal_targets, idx  
    
        
def load_proposal_data(files, orig_data_path, proposal_dir, split):
    image_paths = []
    proposal_coords = []
    image_ids = []
    ground_truths = []

    for file in files:
        image_id = file

        image_path = os.path.join(orig_data_path, "annotated-images", f"img-{image_id}.jpg")

        # Set paths for proposals and ground truths based on split
        proposal_pickle_path = os.path.join(proposal_dir, f'{split}_target_img-{image_id}.pkl')
        ground_truth_path = os.path.join(proposal_dir, f'img-{image_id}_gt.pkl')

        if not os.path.exists(proposal_pickle_path):
            print(f"File not found: {proposal_pickle_path}")
            continue
        if not os.path.exists(ground_truth_path):
            print(f"Ground truth file not found: {ground_truth_path}")
            continue

        try:
            # Load proposals
            with open(proposal_pickle_path, 'rb') as f:
                proposal_data = pk.load(f)

            if isinstance(proposal_data, tuple) and len(proposal_data) >= 2:
                proposal_targets = proposal_data[1]
            else:
                print(f"Unexpected data format in {proposal_pickle_path}")
                continue

            if not proposal_targets:
                print(f"No proposals found for image {image_id}")
                continue

            # Load ground truth data
            with open(ground_truth_path, 'rb') as f:
                ground_truth = pk.load(f)

            coords = []
            for target in proposal_targets:
                if target is None:
                    continue
                try:
                    x_min = int(target['image_xmin'])
                    y_min = int(target['image_ymin'])
                    x_max = int(target['image_xmax'])
                    y_max = int(target['image_ymax'])

                    if x_max > x_min and y_max > y_min:
                        coords.append(TensorDict({
                            'xmin': torch.tensor(x_min, dtype=torch.float32),
                            'ymin': torch.tensor(y_min, dtype=torch.float32),
                            'xmax': torch.tensor(x_max, dtype=torch.float32),
                            'ymax': torch.tensor(y_max, dtype=torch.float32),
                            'original_image_name': image_id
                        }))
                except KeyError as e:
                    print(f"KeyError: {e} in target {target}")
                    continue

            if coords:
                image_paths.append(image_path)
                proposal_coords.append(coords)
                image_ids.append(image_id)
                ground_truths.append(ground_truth)
            else:
                print(f"No valid coordinates for image {image_id}")

        except Exception as e:
            print(f"Error loading proposals for image {image_id}: {e}")

    return image_paths, proposal_coords, image_ids, ground_truths


class ValAndTestDataset(Dataset):
    def __init__(self, base_dir, split='val', transform=None, orig_data_path='Potholes'):
        self.transform = transform
        self.split = split.lower()

        assert split in ["val", "test"], "Split must be either 'val' or 'test'"
        self.proposal_dir = os.path.join(base_dir, f'{self.split}_data', 'targets')

        if not os.path.exists(self.proposal_dir):
            raise FileNotFoundError(f"Directory not found: {self.proposal_dir}")

        proposal_files = glob.glob(os.path.join(self.proposal_dir, f'{self.split}_target_img-*.pkl'))
        self.files = [os.path.basename(file).replace(f'{self.split}_target_img-', '').replace('.pkl', '') for file in proposal_files]


        self.image_paths, self.proposal_coords, self.image_ids, self.ground_truths = load_proposal_data(
            self.files, orig_data_path, self.proposal_dir, split=split
        )

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        coords = self.proposal_coords[idx]
        ground_truth = self.ground_truths[idx]

        with Image.open(image_path) as img:
            original_image = img.convert('RGB')

        cropped_proposals_images = []
        for coord in coords:
            x_min = int(coord['xmin'])
            y_min = int(coord['ymin'])
            x_max = int(coord['xmax'])
            y_max = int(coord['ymax'])
            proposal_image = original_image.crop((x_min, y_min, x_max, y_max))

            if self.transform:
                proposal_image = self.transform(proposal_image)

            cropped_proposals_images.append(proposal_image)

        return original_image, cropped_proposals_images, coords, self.image_ids[idx], ground_truth


def val_test_collate_fn_cropped(batch):
    batch_original_images = []
    batch_proposal_images = []
    batch_coords = []
    batch_image_ids = []
    batch_ground_truths = []

    for original_image, proposal_images, coords, image_id, ground_truth in batch:
        batch_original_images.append(original_image)
        batch_proposal_images.append(proposal_images)
        batch_coords.append(coords)
        batch_image_ids.append(image_id)
        batch_ground_truths.append(ground_truth)

    return batch_original_images, batch_proposal_images, batch_coords, batch_image_ids, batch_ground_truths
def get_xml_data(xml_path):
    
    tree = ET.parse(xml_path)
    root = tree.getroot()

    # Initialize lists for the target
    targets = []

    # Iterate through each object in the XML file
    for obj in root.findall('object'):
    
        # If the box is a pothole the label is 1 (True)
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

        # Append bounding box and label. TensorDict is used to convert the dictionary to a tensor
        directory = TensorDict({
            'xmin'  : torch.tensor(xmin, dtype=torch.float32),
            'ymin'  : torch.tensor(ymin, dtype=torch.float32),
            'xmax'  : torch.tensor(xmax, dtype=torch.float32),
            'ymax'  : torch.tensor(ymax, dtype=torch.float32),
            'labels': torch.tensor(label, dtype=torch.int64)
        })

        targets.append(directory)
    
    return targets
    
def pickle_save(final_image, final_target, save_images_path, save_targets_path, index, split='train' ):
    if split == 'train':
        # Create the directory in the blackhole path if it doesn't exist
        os.makedirs(save_images_path, exist_ok=True)
        os.makedirs(save_targets_path, exist_ok=True)

        # Save the files to the BLACKHOLE path
        with open(os.path.join(save_images_path, f'train_image_{index}.pkl'), 'wb') as f:
            pk.dump(final_image, f)
        with open(os.path.join(save_targets_path, f'train_target_{index}.pkl'), 'wb') as f:
            pk.dump(final_target, f)

    elif split == 'val':
        os.makedirs(save_targets_path, exist_ok=True)

        # Save the files to the BLACKHOLE path
        with open(os.path.join(save_targets_path, f'val_target_{index}.pkl'), 'wb') as f:
            pk.dump(final_target, f)
    elif split == 'test':
        os.makedirs(save_targets_path, exist_ok=True)

        # Save the files to the BLACKHOLE path
        with open(os.path.join(save_targets_path, f'test_target_{index}.pkl'), 'wb') as f:
            pk.dump(final_target, f)


def class_balance(proposal_images, proposal_targets, seed, count):
    # Initialize lists for the proposals and targets
    class_1_proposals = []
    class_1_targets = []
    class_0_proposals = []
    class_0_targets = []
    
    random.seed(seed)
    # Loop through each proposal and target
    for image, target in zip(proposal_images, proposal_targets):
        if int(target['label']) == 1:  # Assuming 'labels' is the correct key
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
    assert len(class_0_proposals_new) == total_class_0_ideal, \
        f"Expected {total_class_0_ideal} class 0 proposals, but got {len(class_0_proposals_new)}"

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
        print(f"No proposals and targets found for the image")
        print(f"- Proposals generated in total: {len(proposal_images)}")
        print(f"- Class 0 proposals: {len(class_0_proposals)}")
        print(f"- Class 1 proposals: {len(class_1_proposals)}")

        return None, None
    
def save_ground_truth(ground_truth_path, original_targets):
    with open(ground_truth_path, 'wb') as f:
        pk.dump(original_targets, f)

import matplotlib.patches as patches

def plot_original_and_crops(original_image, ground_truth, cropped_images, n=5):
    """
    Plots the original image with ground truth bounding boxes and n cropped images.
    
    Parameters:
    - original_image: PIL.Image.Image, the original image.
    - ground_truth: list of TensorDicts, each containing 'xmin', 'ymin', 'xmax', 'ymax', and optionally 'labels'.
    - cropped_images: list of transformed proposal images (torch.Tensor).
    - n: int, number of cropped images to display.
    """
    # Convert PIL image to NumPy array for plotting
    original_image_np = np.array(original_image)
    
    # Determine the number of cropped images to display
    n_crops = min(n, len(cropped_images))
    
    # Create subplots: 1 for original image with ground truth, and n for cropped images
    fig, axes = plt.subplots(1, n + 1, figsize=(15, 5))
    
    # Plot the original image
    axes[0].imshow(original_image_np)
    axes[0].set_title('Original Image with Ground Truth')
    axes[0].axis('off')
    
    # Overlay ground truth bounding boxes
    ax = axes[0]
    for gt in ground_truth:
        try:
            xmin = gt['xmin'].item()
            ymin = gt['ymin'].item()
            xmax = gt['xmax'].item()
            ymax = gt['ymax'].item()
            label = gt['labels'].item() if 'labels' in gt.keys() else None
            
            # Create a Rectangle patch
            rect = patches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, 
                                     linewidth=2, edgecolor='r', facecolor='none')
            # Add the patch to the Axes
            ax.add_patch(rect)
            
            # Optionally, add labels
            #if label is not None:
            #    ax.text(xmin, ymin - 5, f'Label: {label}', 
            #            color='yellow', fontsize=8, 
            #            bbox=dict(facecolor='red', alpha=0.5))
        except Exception as e:
            print(f"Error plotting ground truth box: {e}")
    
    # Plot the cropped images
    for i in range(n_crops):
        crop = cropped_images[i]
        if isinstance(crop, torch.Tensor):
            # Convert tensor to NumPy array
            crop_np = crop.permute(1, 2, 0).numpy()
            # Handle normalization if applied
            # Example: If normalized with mean and std, you might need to unnormalize
            # Assuming no normalization for simplicity
            axes[i + 1].imshow(crop_np)
            axes[i + 1].set_title(f'Crop {i + 1}')
            axes[i + 1].axis('off')
        else:
            print(f"Crop {i + 1} is not a tensor.")
    
    plt.tight_layout()
    plt.savefig('original_ground_truth_and_crops.svg', dpi=300)
    plt.show()



#if __name__ == "__main__":
    #import time
    #from torchvision import transforms

    ## Define transformations
    #transform = transforms.Compose([
        #transforms.Resize((256, 256)),  # Adjusted size for typical CNN input
        #transforms.ToTensor(),
    #])

    ## Initialize the validation dataset
    #val_dataset = Val_and_test_data(
        #transform=transform, 
        #folder_path='Potholes', 
        #blackhole_path=VAL_PROPOSALS
    #)
    
    #print(f"Total images in dataset: {len(val_dataset)}")

    ## Set batch size
    #batch_size = 1  # Since each image can have up to 2000 proposals
    
    ## Initialize DataLoader
    #val_loader = DataLoader(
        #val_dataset,
        #batch_size=batch_size,
        #shuffle=True,
        #collate_fn=val_test_collate_fn_cropped
    #)

    ## Start timer
    #start_time = time.time()

    ## Number of cropped images to display
    #n_crops_to_display = 5  # Adjust as needed

    #try:
        ## Retrieve one batch
        #batch = next(iter(val_loader))
        #original_images, proposal_images_list, coords_list, image_ids, ground_truths = batch

        #print(f"\nBatch contents:")
        #print(f"Number of images: {len(original_images)}")
        #print(f"Image IDs: {image_ids}")
        #print(f"Number of proposals: {[len(proposals) for proposals in proposal_images_list]}")
        #print(f"Number of ground truths: {len(ground_truths)}")
        #print("-" * 50)

        ## Plot the first image in the batch
        #plot_original_and_crops(
            #original_images[0], 
            #ground_truths[0], 
            #proposal_images_list[0], 
            #n=n_crops_to_display
        #)

    #except Exception as e:
        #print(f"An error occurred during processing: {e}")

    ## End timer
    #end_time = time.time()

    ## Calculate elapsed time
    #elapsed_time = end_time - start_time
    #print(f"Total time to run the batch: {elapsed_time:.2f} seconds")
