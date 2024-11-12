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
from selective_search import generate_proposals_and_targets_v_2


# REPLACE BY YOUR OWN PATH 


TRAIN_IMAGE_DIR = '/dtu/blackhole/17/209207/train_proposals/image'
TRAIN_TARGET_DIR = '/dtu/blackhole/17/209207/train_proposals/target'
VAL_PROPOSALS = 'Potholes/val_proposals'



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
    """Load proposal data for validation and testing.
    Args:
        files (list): List of file names
        entire_folder_path (str): Path to dataset folder
        blackhole_path (str): Path to blackhole storage
    Returns:
        tuple: Lists of image paths, proposal coordinates, and image IDs
    """
    # Initialize lists for the image paths, proposal coordinates, and image IDs
    image_paths = []
    proposal_coords = []
    image_ids = []

    for file in files:
        # construct the path to the image and XML file and the pickle file
        image_path = os.path.join(entire_folder_path, "annotated-images", file.replace('.xml', '.jpg'))
        image_id = os.path.basename(image_path).split('.')[0]
        proposal_pickle_path = os.path.join(blackhole_path, f'val_target_{image_id}.pkl')
        #TODO: Loop through the pickle file and load image path and proposal coordinates

        # Skip to the next if the pickle file does not exist
        if not os.path.exists(proposal_pickle_path):
            continue

        # Load the proposals from the pickle file
        try:
            with open(proposal_pickle_path, 'rb') as f:
                proposal_targets = pk.load(f)
                

            # Extract the coordinates from the proposals
            coords = []
            for target in proposal_targets:
                try:
                    x_min = int(target['image_xmin'])
                    y_min = int(target['image_ymin'])
                    x_max = int(target['image_xmax'])
                    y_max = int(target['image_ymax'])

                    # Ensure the coordinates are valid and append to the list
                    if x_max > x_min and y_max > y_min:
                        #coords.append(torch.tensordict({
                        #    'image_xmin': torch.tensor(x_min, dtype=torch.float32),
                        #    'image_ymin': torch.tensor(y_min, dtype=torch.float32),
                        #    'image_xmax': torch.tensor(x_max, dtype=torch.float32),
                        #    'image_ymax': torch.tensor(y_max, dtype=torch.float32)
                        #}))
                        coords.append((x_min, y_min, x_max, y_max))
                except KeyError:
                    continue
            # Append the image path, proposal coordinates, and image ID
            if coords:
                image_paths.append(image_path)
                proposal_coords.append(coords)
                image_ids.append(image_id)
                
        except Exception as e:
            print(f"Error loading proposals for image {image_id}: {e}")
            
    return image_paths, proposal_coords, image_ids

class Val_and_test_data(Dataset):
#    def __init__(self, transform=None, folder_path='Potholes', 
#                 method='quality', max_proposals=2000, blackhole_path='/dtu/blackhole/17/209207/val_proposals'):

    def __init__(self, split='val', val_percent=None, seed=42, transform=None, folder_path='Potholes', 
                 method='quality', max_proposals=2000, blackhole_path='/dtu/blackhole/17/209207/val_proposals'):
        self.transform = transform
        self.method = method
        self.max_proposals = max_proposals
        
        # Ensure the dataset is accessed from the root of the repository 
        base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        entire_folder_path = os.path.join(base_path, folder_path)
        
        with open(os.path.join(entire_folder_path, "splits.json"), 'r') as file:
            splits = json.load(file)
            
        if not os.path.exists(entire_folder_path):
            raise FileNotFoundError(f"Directory not found: {entire_folder_path}")
# #TODO!: LINES 138  151 should be deleted!
        # loading of the training split
        train_files = splits['train']
        random.seed(seed)
        random.shuffle(train_files)
        
        if split.lower() == 'val':
            if val_percent is None:
                raise ValueError('Validation percentage is not set')
            val_count = int(len(sorted(glob.glob(os.path.join(entire_folder_path, "annotated-images/img-*.jpg")))) * val_percent / 100)
            files = train_files[:val_count]
        elif split.lower() == 'test':
            files = splits['test']
        else:
            raise ValueError('Use either val or test to access data')
#       
        self.image_paths, self.proposal_coords, self.image_ids = load_proposal_data(files, entire_folder_path, blackhole_path)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load the original image and its index and convert to RGB
        image_path = self.image_paths[idx]
        original_image = Image.open(image_path).convert('RGB')
        image_id = self.image_ids[idx]
        # Load the proposal coordinates
        coords = self.proposal_coords[idx]

        # List for storing the cropped proposal images    
        cropped_proposals_images = []

        # Crop the proposals from the original image
        for coord in coords:
            x_min, y_min, x_max, y_max = coord
            proposal_image = original_image.crop((x_min, y_min, x_max, y_max))

            # Apply the transform if available (e.g., convert to tensor, resize)
            if self.transform:
                proposal_image = self.transform(proposal_image)

            # Append the cropped proposal image
            cropped_proposals_images.append(proposal_image)


        return original_image, cropped_proposals_images, coords, image_id




#class Val_and_test_data(Dataset):
#    def __init__(self, split='val', val_percent=None, seed=42, transform=None, folder_path='Potholes', iou_upper_limit=0.5, iou_lower_limit=0.5, method='quality', max_proposals=int(2000)):
#        self.all_proposal_images = []
#        self.all_proposal_targets = []
#        self.transform = transform
#        self.iou_upper_limit = iou_upper_limit
#        self.iou_lower_limit = iou_lower_limit
#        self.method = method
#        self.max_proposals = max_proposals
#
#        # Ensure the dataset is accessed from the root of the repository
#        base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
#        entire_folder_path = os.path.join(base_path, folder_path)
#
#        # Load the splits from the JSON file
#        json_path = os.path.join(entire_folder_path, "splits.json")
#        with open(json_path, 'r') as file:
#            splits = json.load(file)
#
#        if not os.path.exists(entire_folder_path):
#            raise FileNotFoundError(f"Directory not found: {entire_folder_path}")
#
#        train_files = splits['train']
#        random.seed(seed)
#        random.shuffle(train_files)
#
#        if split.lower() == 'val':
#            if val_percent is not None:
#                number_of_all_files = len(sorted(glob.glob(os.path.join(entire_folder_path, "annotated-images/img-*.jpg"))))
#                val_count = int(number_of_all_files * val_percent / 100)
#                new_val_files = train_files[:val_count]
#                
#                # Print the validation image IDs
#                val_image_ids = [file.replace('.xml', '') for file in new_val_files]
#                print("Validation Image IDs:", val_image_ids)
#                print(f"Number of validation images: {len(new_val_files)}")
#
#                # Set up paths for images and XML files
#                image_paths = [os.path.join(entire_folder_path, "annotated-images", file.replace('.xml', '.jpg')) for file in new_val_files]
#                xml_paths = [os.path.join(entire_folder_path, "annotated-images", file) for file in new_val_files]
#            else:
#                raise ValueError('Validation percentage is not set')
#        elif split.lower() == 'test':
#            image_paths = [os.path.join(entire_folder_path, "annotated-images", file.replace('.xml', '.jpg')) for file in splits['test']]
#            xml_paths = [os.path.join(entire_folder_path, "annotated-images", file) for file in splits['test']]
#        else:
#            raise ValueError('Use either val or test to access data')
#
#        assert len(image_paths) == len(xml_paths), "The length of images and xml files is not the same"
#
#
#
#        for image_path in image_paths:
#            original_image = Image.open(image_path).convert('RGB')
#            image_id = os.path.basename(image_path).split('.')[0]
#            
#            # Generate proposals 
#            _, proposal_targets = generate_proposals_and_targets_copy(
#                original_image, None, self.transform, 
#                None, self.iou_upper_limit, self.iou_lower_limit, 
#                self.method, self.max_proposals, generate_target=False, return_images=False
#            )
#            
#            # proposals and targets 
#            #self.all_proposal_images.extend(proposal_images)
#            self.all_proposal_targets.extend(proposal_targets)
#            
#            # pickle vals 
#            pickle_save_val(self.all_proposal_targets,  train=True, index=image_id)
#            
#            # Rinse and repeat 
#            #self.all_proposal_images = []
#            self.all_proposal_targets = []


#
#    def __len__(self):
#        return len(self.all_proposal_images)
#
#    def __getitem__(self, idx):
#        image_path = self.image_paths[idx]
#        original_image = Image.open(image_path).convert('RGB')
#        coords = self.proposal_coords[idx]
#        proposal_images = []
#
#        for coord in coords:
#            x_min, y_min, x_max, y_max = coord
#            proposal_image = original_image.crop((x_min, y_min, x_max, y_max))
#
#            if self.transform:
#                proposal_image = self.transform(proposal_image)
#
#            proposal_images.append(proposal_image)
#
#        # Optionally, return the image ID for tracking
#        image_id = self.image_ids[idx]
#
#        return proposal_images, coords, image_id



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

def pickle_save(final_image, final_target, save_images_path, save_targets_path, train, index):
    if train:
        # Create the directory in the blackhole path if it doesn't exist
        os.makedirs(save_images_path, exist_ok=True)
        os.makedirs(save_targets_path, exist_ok=True)

        # Save the files to the BLACKHOLE path
        with open(os.path.join(save_images_path, f'train_image_{index}.pkl'), 'wb') as f:
            pk.dump(final_image, f)
        with open(os.path.join(save_targets_path, f'train_target_{index}.pkl'), 'wb') as f:
            pk.dump(final_target, f)

def pickle_save_val(final_target, train, index):
    if train:
        # Use the BLACKHOLE environment variable for flexibility, ensuring absolute path
        blackhole_path = os.getenv('BLACKHOLE', '/dtu/blackhole/17/209207')
        #folder_path_image = os.path.join(blackhole_path, 'val_proposals/image')  # This is an absolute path now
        folder_path_target = os.path.join(blackhole_path, 'val_proposals/target')  # This is an absolute path now

        # Create the directory in the blackhole path if it doesn't exist
        #os.makedirs(folder_path_image, exist_ok=True)
        os.makedirs(folder_path_target, exist_ok=True)

        # Save the files to the BLACKHOLE path
        #with open(os.path.join(folder_path_image, f'val_image_{index}.pkl'), 'wb') as f:
        #    pk.dump(final_image, f)
        with open(os.path.join(folder_path_target, f'val_target_{index}.pkl'), 'wb') as f:
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
    val_dataset = Val_and_test_data(split='val', val_percent=20, transform=transform, folder_path='Potholes', blackhole_path='/dtu/blackhole/17/209207/val_proposals')
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

#val_loader = DataLoader(
#    val_dataset,
#    batch_size=4,
#    shuffle=False,
#    num_workers=2,
#    collate_fn=val_test_collate_fn
#)
#
#print("Printing first few batches:")
#for batch_idx, (cropped_images, coords, image_ids) in enumerate(val_loader):
#    print(f"\nBatch {batch_idx + 1}:")
#    print(f"Number of cropped images: {len(cropped_images)}")
#    print(f"Number of coordinates: {len(coords)}")
#    print(f"Image IDs: {image_ids}")
#    
#    if batch_idx >= 2:  # Print first 3 batches only
#        break
#    print("\nValidation Dataset Information:")
#    print(f"Total number of samples: {len(val_dataset)}")
#    import matplotlib.pyplot as plt
#    import matplotlib.patches as patches
#
#    def plot_image_with_proposals(dataset, image_idx, num_proposals=10):
#        # Get the image and proposals
#        image, coords, idx = dataset[image_idx]
#        
#        # Convert image tensor to numpy for plotting
#        image_np = image.permute(1, 2, 0).numpy()
#        
#        # Create figure and axes
#        fig, ax = plt.subplots(1)
#        ax.imshow(image_np)
#        
#        # Plot the first n proposals
#        for i, coord in enumerate(coords[:num_proposals]):
#            x_min, y_min, x_max, y_max = coord
#            width = x_max - x_min
#            height = y_max - y_min
#            
#            # Create a rectangle patch
#            rect = patches.Rectangle((x_min, y_min), width, height, 
#                                   linewidth=2, edgecolor='r', facecolor='none')
#            ax.add_patch(rect)
#        
#        plt.title(f'Image with {num_proposals} proposals')
#        plt.savefig(f'pothole_proposals_{image_idx}.png')
#        plt.show()

    # Example usage
#    n_proposals = 100  # Change this number to show different number of proposals
#    image_idx = 3    # Change this to view different images
#    plot_image_with_proposals(val_dataset, image_idx, n_proposals)
    # Iterate through multiple batches
#    num_batches_to_print = 5  # Print first 5 batches
#    
#    for batch_idx, (original_images, all_proposal_coords, image_indices) in enumerate(val_loader):
#        if batch_idx >= num_batches_to_print:
#            break
#            
#        print(f"\nBatch {batch_idx + 1}:")
#        print(f"Original Images shape: {original_images.shape}")
#        print(f"Number of proposal coordinates: {len(all_proposal_coords)}")
#        print(f"Number of image indices: {len(image_indices)}")
#        
#        # Print example coordinates from first few proposals
#        print("\nExample proposals from this batch:")
#        for i, (img_idx, coord) in enumerate(all_proposal_coords[:3]):  # Show first 3 proposals
#            print(f"Proposal {i+1}: Image Index {img_idx}, Coordinates {coord}")
#        
#        print("-" * 50)  # Separator between batches


#    for batch_idx, (original_images, all_proposal_coords, _) in enumerate(val_loader):
#        proposal_images = []
#        proposal_image_indices = []
#
#        for (img_idx, coord) in all_proposal_coords:
#            x_min, y_min, x_max, y_max = coord
#
#            # Get the dimensions of the image
#            _, H, W = original_images[img_idx].shape
#
#            # Convert coordinates to integers
#            x_min = int(x_min)
#            y_min = int(y_min)
#            x_max = int(x_max)
#            y_max = int(y_max)
#
#            # Clamp coordinates to be within the image dimensions
#            x_min = max(0, min(x_min, W - 1))
#            x_max = max(0, min(x_max, W))
#            y_min = max(0, min(y_min, H - 1))
#            y_max = max(0, min(y_max, H))
#
#            # Ensure the coordinates result in a valid crop
#            if x_max <= x_min or y_max <= y_min:
#                print(f"Skipping proposal with invalid dimensions: x_min={x_min}, x_max={x_max}, y_min={y_min}, y_max={y_max}")
#                continue
#
#            # Get the corresponding image
#            original_image = original_images[img_idx]
#
#            # Crop the proposal
#            proposal_image = original_image[:, y_min:y_max, x_min:x_max]
#
#            # Check if the proposal_image is non-empty
#            if proposal_image.numel() == 0:
#                print(f"Empty proposal after cropping: x_min={x_min}, x_max={x_max}, y_min={y_min}, y_max={y_max}")
#                continue
#
#            # Resize if necessary
#            if resize_transform:
#                proposal_image = resize_transform(proposal_image)
#
#            proposal_images.append(proposal_image)
#            proposal_image_indices.append(img_idx)
#
#        # Stack proposals into a tensor
#        if proposal_images:
#            proposal_tensors = torch.stack(proposal_images)
#        else:
#            continue  # Skip if no proposals
#
#        # Move to device
#        proposal_tensors = proposal_tensors.to(device)
#
#        # Forward pass through the model
#        # outputs = model(proposal_tensors)
#
#        # Map outputs back to images using proposal_image_indices
#        # ...
#
#        # Optionally, break after first batch for testing
#        break








    # Now, you have outputs for all proposals in the batch
    # You can process them and map back to the original images using proposal_image_indices


    
    #val = Val_and_test_data(split='val',val_percent=20,  transform=transform, folder_path='Potholes')
    #for i in range(0, len(val)):
    #    print(val[i])
    #print(f"Validation samples: {len(val)}")
    #test = Val_and_test_data(split='test',transform=transform, folder_path='Potholes')

    #train_files = set(train)
    #val_files = set(val)
    #test_files = set(test)

    #print(f"Training samples: {len(train)}")
    #print(f"Validation samples: {len(val)}")
    #print(f"Test samples: {len(test)}")
    #assert train.isdisjoint(val_files), "Train and validation sets overlap"
    #assert val.isdisjoint(test_files), "Train and test sets overlap"
    #assert test.isdisjoint(test_files), "Validation and test sets overlap"


    #dataloader = DataLoader(train, batch_size = 1, shuffle=True, num_workers=0)
    #end_time = time.time()

    #print("Time taken to load one batch:", end_time - start_time, "seconds")

    # print out the first batch

#    for batch_idx, (proposal_images, proposal_targets) in enumerate(dataloader):
#        print(f"\nBatch {batch_idx + 1}:")
#        print(f"Proposal Images: {len(proposal_images)}")
#        print(f"Proposal Targets: {len(proposal_targets)}")
#
#        # Print details of the first image in the batch as an example
#        print("\nExample from the batch:")
#        print(f"Proposal Image Shape: {proposal_images[0].size}")
#        print(f"Proposal Target: {proposal_targets[0]}")
#        break
    # print type of train 
    #print(type(train))



    #start_time = time.time()
    #val = Val_and_test_data(split='test', val_percent=20, transform=transform, folder_path='Potholes')
    #dataloader_val = DataLoader(val, batch_size = 8, shuffle=True, num_workers=8, collate_fn=val_test_collate_fn)
    #end_time = time.time()
    
    #print("Time taken to load one batch:", end_time - start_time, "seconds")
    #count = 0
    #print('check')

#    for batch_idx, (original_images, original_targets, proposal_images, proposal_targets) in enumerate(dataloader_val):
#
#        print(f"\nBatch {batch_idx + 1}:")
#        print(f"Original Images: {len(original_images)}")
#        print(f"Original Targets: {len(original_targets)}")
#        print(f"Proposal Images: {len(proposal_images)}")
#        print(f"Proposal Targets: {len(proposal_targets)}")
#
#        # Print details of the first image in the batch as an example
#        print("\nExample from the batch:")
#        print(f"Original Image Shape: {original_images[0].size}")
#        print(f"Original Target: {original_targets[0]}")
#        print(f"Number of Proposals: {len(proposal_images[0])}")
#    #    print(f"Proposal Target Example: {proposal_targets[0][:5]}")  # Print the first few proposals
#
#    #    # Stop after printing one batch (remove this break to print all batches)
#        break
#
#        #visualize_samples(dataloader, num_images=4, figname='pothole_samples', box_thickness=5)