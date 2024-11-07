import os
import glob
import time
import torch
import json
import xml.etree.ElementTree as ET
import random

from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tensordict import TensorDict
from visualize import visualize_samples, visualize_proposal
import json
from torch.utils.data import default_collate
from selective_search_new import  generate_proposals_and_targets
from torch.utils.data import default_collate
import pickle as pk



class Proposals(Dataset):
    def __init__(self, split='train', val_percent=None, seed=42, transform=None, folder_path='Potholes'):

        #TODO make a check, if the file already exist, then just take the file from there, else generate new
        #TODO ADD upper and lower limit as input to the class 
        # TODO implement class imbalance 

        #final_image, final_target
        #try 
#
        #self.final_imges, self.final target = open folder
#
        #else: 
        
        #The to lists contains a number of proposals and targets that is going to be used in training 
        self.all_proposal_images = []
        self.all_proposal_targets = []
    
        # Ensure the dataset is accessed from the root of the repository
        base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        entire_folder_path = os.path.join(base_path, folder_path)

        # Check if the folder path exists
        if not os.path.exists(entire_folder_path):
            raise FileNotFoundError(f"Directory not found: {entire_folder_path}")

        # Load the splits from the JSON file
        json_path = os.path.join(entire_folder_path, "splits.json")
        with open(json_path, 'r') as file:
            splits = json.load(file)

        #If the validation percentage for the split is set, it will create a validation set based on the existing training set
        if val_percent is not None:
            #Get all the files to calculate the precentage for validation set
            number_of_all_files = len(sorted(glob.glob(os.path.join(entire_folder_path, "annotated-images/img-*.jpg")))) #Get the number of all the files in the folder 
            train_files = splits['train']
            random.seed(seed)
            random.shuffle(train_files)  

            # Calculate the number of validation samples
            val_count = int(number_of_all_files * val_percent/100)
            new_val_files = train_files[:val_count]
            new_train_files = train_files[val_count:]

            image_paths = [os.path.join(entire_folder_path, "annotated-images", file.replace('.xml', '.jpg')) for file in new_train_files]
            xml_paths = [os.path.join(entire_folder_path, "annotated-images", file) for file in new_train_files]



            # counte to keep track of the number of images
            count = 0

            # storing of a class proposals and targets 

            # looping over the images and xml files
            for image_path, xml_path in zip(image_paths, xml_paths):

                # Initialize lists for the proposals and targets
                class_0_proposals = []
                class_0_targets = []
                class_1_proposals = []
                class_1_targets = []

                original_image = Image.open(image_path).convert('RGB')
                original_targets = get_xml_data(xml_path)
                # Get the image id for pickling ids
                image_id = image_path.split('/')[-1].split('.')[0]


                # print out every 50th image id for tracking while running
                if count % 50 == 0:
                    print(f"Image id {count}")

                
                # Generate proposals and targets
                proposal_images, proposal_targets = generate_proposals_and_targets(original_image, original_targets, transform)
                print(f"Number of proposals generated for image {image_id}: {len(proposal_images)}")
                #print(proposal_images)
            
                # Loop through each proposal and target
                for image, target, in zip(proposal_images, proposal_targets):
                    if int(target['label']) == 1:
                        # print if label is 1 missing and write the image id
                        class_1_proposals.append(image)
                        class_1_targets.append(target)
                    else:
                        class_0_proposals.append(image)
                        class_0_targets.append(target)                
                #count += 1
                #print(count)
                #if count > len(new_train_files):
                #    if count % 50 == 0:
                #        print(f"Image id {image_id}")
                #    break


                # Class balancing
                total_class_1 = len(class_1_proposals)   # 25 % of the class 0 proposals
                total_class_0_ideal = int(total_class_1 * 3)     # 75 % of the class 0 proposals 
                total_class_0 = len(class_0_proposals)

                # sanity check 
                #print(f"Total class 1: {total_class_1}")
                #print(f"Total class 0: {total_class_0}")
                #print(f"Total class 0 should have after class balancing: {total_class_0_ideal}")
                #pickle_save(self.all_proposal_images, self.all_proposal_targets, train=True)
                #self.final_image, self.final_target = class_imbalance(all_proposal_images, all_proposal_target)

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

                    # Append the proposals and targets to the final lists
                    self.all_proposal_images.extend(image_proposals)
                    self.all_proposal_targets.extend(image_targets)
                    #print("*" * 40)
                    # print he porposals and targets at class 0 
                    #print(f"Class 0 proposals before class balancing: {len(class_0_proposals)}")
                    #print(f"Class 0 targets before class balancing: {len(class_0_targets)}")
                    #print("***************** Balancing now *****************")
                    #print(f"Class 0 proposals after class balancing: {len(class_0_proposals_new)}")
                    #print(f"Class 0 targets after class balancing: {len(class_0_targets_new)}")
                    pickle_save(self.all_proposal_images, self.all_proposal_targets, train=True, index=image_id)

                    # Rinse and repeat for the next image
                    self.all_proposal_images = []
                    self.all_proposal_targets = []
                    count += 1
                else:
                    print(f"No proposals and targets found for the image {image_id}")
                    print(f"- Proposals generated in total: {len(proposal_images)}")
                    print(f"- Class 0 proposals: {len(class_0_proposals)}")
                    print(f"- Class 1 proposals: {len(class_1_proposals)}")
                    continue



    def __len__(self):
        """Returns the total number of samples in the dataset."""
        return len(self.all_proposal_images)

    def __getitem__(self, idx):
        #return self.final_image[idx], self.final_target[idx]
        return self.all_proposal_images[idx], self.all_proposal_targets[idx]



def pickle_save(final_image, final_target, train, index):
    if train:
        # Use the BLACKHOLE environment variable for flexibility, ensuring absolute path
        blackhole_path = os.getenv('BLACKHOLE', '/dtu/blackhole/17/209207')
        folder_path_image = os.path.join(blackhole_path, 'train_proposals/image')  # This is an absolute path now
        folder_path_target = os.path.join(blackhole_path, 'train_proposals/target')  # This is an absolute path now

        # Create the directory in the blackhole path if it doesn't exist
        os.makedirs(folder_path_image, exist_ok=True)
        os.makedirs(folder_path_target, exist_ok=True)

        # Save the files to the BLACKHOLE path
        with open(os.path.join(folder_path_image, f'train_image_{index}.pkl'), 'wb') as f:
            pk.dump(final_image, f)
        with open(os.path.join(folder_path_target, f'train_target_{index}.pkl'), 'wb') as f:
            pk.dump(final_target, f)



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

def custom_collate_fn(batch):
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

    batch_images = []
    batch_targets = []

    for image, target in batch:
        batch_images.append(image)  # Append the image part to the images list
        batch_targets.append(target)  # Append the target part to the targets list
    return default_collate(batch_images), batch_targets  # Return stacked images and original targets




if __name__ == "__main__":
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])
    prop = Proposals(split = 'Train', val_percent=20, transform=transform, folder_path='Potholes')

    #dataloader = DataLoader(prop, batch_size=32, shuffle=True, num_workers=8, collate_fn=custom_collate_fn)
    #visualize_samples(dataloader, num_images=4, figname='pothole_samples', box_thickness=5)
    #visualize_proposal(prop.all_proposal_images[400], prop.all_proposal_targets[0], box_thickness=2, figname='proposals.png')


#    for batch_images, batch_targets in dataloader:
#        print("Batch images shape:", batch_images.shape)  # Should print: (2, 3, 256, 256)
#
#        count = 0
#        # Iterate through the targets to show correspondence with images
#        for image, target in enumerate(batch_targets):            
#
#            if int(target['label']) == 1:
#                print(f"Image {image} has {len(target)} target(s):")
#                print('length of target', len(target))
#                print(type(target['gt_bbox_xmin']))
#                print(f"  Bounding box: {target['gt_bbox_xmin']:.3g}, {target['gt_bbox_ymin']:.3g}, {target['gt_bbox_xmax']:.3g}, {target['gt_bbox_ymax']:.3g}, Label: {target['label']}")
#                count += 1
#            if count >= 5:
#                break # only show 5 images
#        break         # Only show one batch

    #print(prop.image_proposals_list[1].shape)
    #for image, target in zip(prop.image_proposals_list, prop.target_proposals_list):
    #    if int(target['label']) == 1:
    #        print(target)
    #        visualize_proposal(image, target, box_thickness=1, figname='proposals.png')
    #        break
    #
    #print('hello')