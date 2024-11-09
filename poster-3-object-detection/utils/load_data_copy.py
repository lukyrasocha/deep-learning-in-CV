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
from selective_search import generate_proposals_and_targets


# REPLACE BY YOUR OWN PATH 
IMAGE_DIR = '/dtu/blackhole/17/209207/train_proposals/image'
TARGET_DIR = '/dtu/blackhole/17/209207/train_proposals/target'


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
        return len(self.image_path)
    
    def __getitem__(self, idx):
        image = self.image_path[idx]
        target = self.target_path[idx]

        with open(image, 'rb') as f:
            proposal_image = pk.load(f)

        with open(target, 'rb') as f:
            proposal_target = pk.load(f)

        return proposal_image, proposal_target
        

class Val_and_test_data(Dataset):
    def __init__(self, split='val', val_percent=None, seed=42, transform=None, folder_path='Potholes', iou_upper_limit=0.5, iou_lower_limit=0.5, method='quality', max_proposals=int(2000)):
        self.all_proposal_images = []
        self.all_proposal_targets = []
        self.transform = transform
        self.iou_upper_limit = iou_upper_limit
        self.iou_lower_limit = iou_lower_limit
        self.method = method
        self.max_proposals = max_proposals

        # Ensure the dataset is accessed from the root of the repository
        base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        entire_folder_path = os.path.join(base_path, folder_path)

        # Load the splits from the JSON file
        json_path = os.path.join(entire_folder_path, "splits.json")
        with open(json_path, 'r') as file:
            splits = json.load(file)

        if not os.path.exists(entire_folder_path):
            raise FileNotFoundError(f"Directory not found: {entire_folder_path}")

        train_files = splits['train']
        random.seed(seed)
        random.shuffle(train_files)

        if split.lower() == 'val':
            if val_percent is not None:
                number_of_all_files = len(sorted(glob.glob(os.path.join(entire_folder_path, "annotated-images/img-*.jpg"))))
                val_count = int(number_of_all_files * val_percent / 100)
                new_val_files = train_files[:val_count]
                
                # Print the validation image IDs
                val_image_ids = [file.replace('.xml', '') for file in new_val_files]
                print("Validation Image IDs:", val_image_ids)
                print(f"Number of validation images: {len(new_val_files)}")

                # Set up paths for images and XML files
                image_paths = [os.path.join(entire_folder_path, "annotated-images", file.replace('.xml', '.jpg')) for file in new_val_files]
                xml_paths = [os.path.join(entire_folder_path, "annotated-images", file) for file in new_val_files]
            else:
                raise ValueError('Validation percentage is not set')
        elif split.lower() == 'test':
            image_paths = [os.path.join(entire_folder_path, "annotated-images", file.replace('.xml', '.jpg')) for file in splits['test']]
            xml_paths = [os.path.join(entire_folder_path, "annotated-images", file) for file in splits['test']]
        else:
            raise ValueError('Use either val or test to access data')

        assert len(image_paths) == len(xml_paths), "The length of images and xml files is not the same"



        for image_path in image_paths:
            original_image = Image.open(image_path).convert('RGB')
            image_id = os.path.basename(image_path).split('.')[0]
            
            # Generate proposals 
            proposal_images, proposal_targets = generate_proposals_and_targets(
                original_image, None, self.transform, 
                None, self.iou_upper_limit, self.iou_lower_limit, 
                self.method, self.max_proposals, generate_target=False
            )
            
            # proposals and targets 
            self.all_proposal_images.extend(proposal_images)
            self.all_proposal_targets.extend(proposal_targets)
            
            # pickle vals 
            pickle_save_val(self.all_proposal_images, self.all_proposal_targets, train=True, index=image_id)
            
            # Rinse and repeat 
            self.all_proposal_images = []
            self.all_proposal_targets = []



    def __len__(self):
        return len(self.all_original_images)

    def __getitem__(self, idx):
        original_image = self.all_original_images[idx]
        original_targets = self.all_original_targets[idx]

        # Generate proposals and targets on the fly
        #proposal_images, proposal_targets = generate_proposals_and_targets(
        #    original_image, original_targets, self.transform, 
        #    None, self.iou_upper_limit, self.iou_lower_limit, 
        #    self.method, self.max_proposals, generate_target=False
        #)

        #original_image = self.transform(original_image)

        return original_image, original_targets


def val_test_collate_fn(batch):
    """
    Custom collate function for a PyTorch DataLoader, used to process a batch of data 
    where each sample contains an image and its corresponding targets.
    It is needed because the deafult collate function expect dimensions of same size

    Parameters:
        batch (list of tuples): A list where each element is a tuple (image, target).
            - image: A tensor representing the image.
            - target: A list or dictionary containing the bounding box coordinates 
                      and labels for objects detected in the image.

    Returns:
        tuple: A tuple containing:
            - images (Tensor): A stacked tensor of images with shape 
              (batch_size, channels, height, width), created using PyTorch's `default_collate`.
            - targets (list): A list of original target annotations, one for each image.
    """

    batch_original_images = []
    batch_original_targets = []
    batch_proposal_images = []
    batch_proposal_targets = []

    for original_image, original_target, proposal_images, proposal_targets in batch:
        batch_original_images.append(original_image)  # Append the image part to the images list
        batch_original_targets.append(original_target)  # Append the target part to the targets list
        batch_proposal_images.append(proposal_images)
        batch_proposal_targets.extend(proposal_targets)
    
    return default_collate(batch_original_images), batch_original_targets, default_collate(batch_proposal_images), batch_proposal_targets  # Return stacked images and original targets

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

def pickle_save_val(final_image, final_target, train, index):
    if train:
        # Use the BLACKHOLE environment variable for flexibility, ensuring absolute path
        blackhole_path = os.getenv('BLACKHOLE', '/dtu/blackhole/17/209207')
        folder_path_image = os.path.join(blackhole_path, 'val_proposals/image')  # This is an absolute path now
        folder_path_target = os.path.join(blackhole_path, 'val_proposals/target')  # This is an absolute path now

        # Create the directory in the blackhole path if it doesn't exist
        os.makedirs(folder_path_image, exist_ok=True)
        os.makedirs(folder_path_target, exist_ok=True)

        # Save the files to the BLACKHOLE path
        with open(os.path.join(folder_path_image, f'val_image_{index}.pkl'), 'wb') as f:
            pk.dump(final_image, f)
        with open(os.path.join(folder_path_target, f'val_target_{index}.pkl'), 'wb') as f:
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

if __name__ == "__main__":
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])


    ###############################################################
    #The following code is use for the validation/test dataloader
    ###############################################################


    # Load the train data 
    #start_time = time.time()

    #train = Trainingset(IMAGE_DIR, TARGET_DIR)
    val = Val_and_test_data(split='val',val_percent=20,  transform=transform, folder_path='Potholes')
    print(f"Validation samples: {len(val)}")
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