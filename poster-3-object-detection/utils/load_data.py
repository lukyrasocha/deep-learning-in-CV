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
from torch.utils.data import default_collate
from utils.selective_search import generate_proposals_for_test_and_val


# REPLACE BY YOUR OWN BLACKHOLE 
TRAIN_IMAGE_DIR = '/dtu/blackhole/1b/209339/training_data/images/'
TRAIN_TARGET_DIR = '/dtu/blackhole/1b/209339/training_data/targets/'
VAL_PROPOSALS = '/dtu/blackhole/1b/209339/validation_data/targets/'
TEST_PROPOSALS = '/dtu/blackhole/1b/209339/test_data/targets/'



#Implement the one below
class Trainingset(Dataset):
    def __init__(self, image_path, target_path):
        self.image_path = image_path
        self.target_path = target_path
        
        # Accessing from the root directory
        self.image_path = glob.glob(os.path.join(self.image_path, '*.pkl'))
        self.target_path = glob.glob(os.path.join(self.target_path, '*.pkl'))

        assert len(self.image_path) == len(self.target_path), "The length of images and targets is not the same"

    def __len__(self):
        return len(self.image_paths)

    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        original_image = Image.open(image_path).convert('RGB')

        if self.transform:
            original_image = self.transform(original_image)

        coords = self.proposal_coords[idx]

        # Return the original image, proposal coordinates, and index
        return original_image, coords, idx
    

    

def load_proposal_data(files, entire_folder_path, blackhole_path):
    image_paths = []
    proposal_coords = []
    image_ids = []

    for file in files:
        image_id = file  # Now 'image_id' is '12'

        image_path = os.path.join(entire_folder_path, "annotated-images", f"img-{image_id}.jpg")
        if blackhole_path == VAL_PROPOSALS:
            proposal_pickle_path = os.path.join(blackhole_path, f'val_target_img-{image_id}.pkl')
        elif blackhole_path == TEST_PROPOSALS:
            proposal_pickle_path = os.path.join(blackhole_path, f'test_target_img-{image_id}.pkl')

        if not os.path.exists(proposal_pickle_path):
            print(f"File not found: {proposal_pickle_path}")
            continue

        try:
            with open(proposal_pickle_path, 'rb') as f:
                proposal_data = pk.load(f)

            # Handle the case where proposal_data is a tuple (None, proposals)
            if isinstance(proposal_data, tuple) and len(proposal_data) >= 2:
                proposal_targets = proposal_data[1]
            else:
                print(f"Unexpected data format in {proposal_pickle_path}")
                continue

            if not proposal_targets:
                print(f"No proposals found for image {image_id}")
                continue

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
                        coords.append((x_min, y_min, x_max, y_max))
                except KeyError as e:
                    print(f"KeyError: {e} in target {target}")
                    continue

            if coords:
                image_paths.append(image_path)
                proposal_coords.append(coords)
                image_ids.append(image_id)
            else:
                print(f"No valid coordinates for image {image_id}")

        except Exception as e:
            print(f"Error loading proposals for image {image_id}: {e}")

    return image_paths, proposal_coords, image_ids


class Val_and_test_data(Dataset):
    def __init__(self, transform=None, folder_path='Potholes', 
                 method='quality', max_proposals=2000, blackhole_path=VAL_PROPOSALS):
        self.transform = transform
        self.method = method
        self.max_proposals = max_proposals

        base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        entire_folder_path = os.path.join(base_path, folder_path)
            
        if not os.path.exists(entire_folder_path):
            raise FileNotFoundError(f"Directory not found: {entire_folder_path}")

        if blackhole_path == VAL_PROPOSALS:
            pickled_files = glob.glob(os.path.join(blackhole_path, 'val_target_img-*.pkl'))
            # Correctly extract the image ids from the pickled files 
            files = [os.path.basename(file).replace('val_target_img-', '').replace('.pkl', '') for file in pickled_files]
            
        elif blackhole_path == TEST_PROPOSALS:
            pickled_files = glob.glob(os.path.join(blackhole_path, 'test_target_img-*.pkl'))
            # Correctly extract the image ids from the pickled files
            files = [os.path.basename(file).replace('test_target_img-', '').replace('.pkl', '') for file in pickled_files]
            

        
        self.image_paths, self.proposal_coords, self.image_ids = load_proposal_data(files, entire_folder_path, blackhole_path)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        original_image = Image.open(image_path).convert('RGB')
        image_id = self.image_ids[idx]
        coords = self.proposal_coords[idx]

        cropped_proposals_images = []

        for coord in coords:
            x_min, y_min, x_max, y_max = coord
            proposal_image = original_image.crop((x_min, y_min, x_max, y_max))

            if self.transform:
                proposal_image = self.transform(proposal_image)

            cropped_proposals_images.append(proposal_image)

        return original_image, cropped_proposals_images, coords, image_id





def val_test_collate_fn_cropped(batch):
    """
    Custom collate function to handle variable numbers of proposals per image.
    """

    batch_proposal_images = []
    batch_coords = []
    batch_image_ids = []

    for proposal_images, coords, image_id in batch:
        batch_proposal_images.extend(proposal_images)
        batch_coords.extend(coords)
        # List where the imamge id is repeated for each proposal image
        # for example if there are 3 proposals for an image, the image id will be repeated 3 times
        batch_image_ids.extend([image_id] * len(proposal_images))

    # Stack the proposal images into a tensor to ensure the 
    # [batch_size, 3, H, W] shape is maintained
    if batch_proposal_images:  
        batch_proposal_images = torch.stack(batch_proposal_images)
    else:
        batch_proposal_images = torch.Tensor()

    return batch_proposal_images, batch_coords, batch_image_ids


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

def val_test_collate_fn_cropped(batch):
    """
    Custom collate function to handle variable numbers of proposals per image.
    """
    batch_original_images = []
    batch_proposal_images = []
    batch_coords = []
    batch_image_ids = []

    for original_image, proposal_images, coords, image_id in batch:
        batch_original_images.append(original_image)
        batch_proposal_images.append(proposal_images)
        batch_coords.append(coords)
        batch_image_ids.append(image_id)

    return batch_original_images, batch_proposal_images, batch_coords, batch_image_ids

        
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
    
def plot_original_and_crops(original_image, cropped_images, n=5):
    """
    Plots the original image and n cropped images.
    """
    # Convert PIL image to numpy array for plotting
    original_image_np = np.array(original_image)

    # Determine the number of cropped images to display
    n_crops = min(n, len(cropped_images))

    # Create subplots
    fig, axes = plt.subplots(1, n_crops + 1, figsize=(15, 5))

    # Plot the original image
    axes[0].imshow(original_image_np)
    axes[0].set_title('Original Image')
    axes[0].axis('off')

    # Plot the cropped images
    for i in range(n_crops):
        crop_np = cropped_images[i].permute(1, 2, 0).numpy()  # Convert tensor to numpy array
        # If the images were normalized during transform, you might need to unnormalize them
        axes[i + 1].imshow(crop_np)
        axes[i + 1].set_title(f'Crop {i + 1}')
        axes[i + 1].axis('off')

    plt.tight_layout()
    plt.savefig('original_and_crops.svg', dpi=300)
    plt.show()
if __name__ == "__main__":



    import time
    transform = transforms.Compose([
        transforms.Resize((256, 256)),  # Adjusted size for typical CNN input
        transforms.ToTensor(),
    ])
    val_dataset = Val_and_test_data(transform=transform, folder_path='Potholes', blackhole_path=TEST_PROPOSALS)
    #val_dataset = Val_and_test_data(transform=transform, folder_path='Potholes', blackhole_path='/dtu/blackhole/17/209207/val_proposals')
    print(f"Total images in dataset: {len(val_dataset)}")

    # Adjust batch_size to manage memory usage
    batch_size = 1  # Since each image can have up to 2000 proposals
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        collate_fn=val_test_collate_fn_cropped
    )

    # Start timer
    start_time = time.time()

    # Number of cropped images to display
    n_crops_to_display = 15  # You can set this to any number you like

    # Process the validation set
    for batch_idx, (original_images, proposal_images_list, coords_list, image_ids) in enumerate(val_loader):
        print(f"\nBatch {batch_idx + 1}:")
        print(f"Number of images in batch: {len(original_images)}")
        print(f"Image IDs: {image_ids}")
        print("-" * 50)  # Separator between batches

        # Since batch_size=1, we'll work with the first (and only) item
        original_image = original_images[0]
        #print(len(original_image))
        proposal_images = proposal_images_list[0]
        print(len(proposal_images))
        coords = coords_list[0]
        image_id = image_ids[0]

        # Plot the original image and n cropped images
        plot_original_and_crops(original_image, proposal_images, n=n_crops_to_display)

        # If you wish to process the proposal_images through your model, you can stack them
        proposal_images_tensor = torch.stack(proposal_images)  # Shape: [num_proposals, 3, H, W]
        # outputs = model(proposal_images_tensor)

        # For demonstration, let's just simulate processing time
        time.sleep(0.1)  # Simulate computation
        break
        # Break after first batch for demonstration purposes
        

    # End timer
    end_time = time.time()

    # Calculate elapsed time
    elapsed_time = end_time - start_time
    print(f"Total time to run the batch: {elapsed_time:.2f} seconds")
