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

            count = 0

            for image_path, xml_path in zip(image_paths, xml_paths):
                original_image = Image.open(image_path).convert('RGB')
                original_targets = get_xml_data(xml_path)

                proposal_images, proposal_targets = generate_proposals_and_targets(original_image, original_targets, transform)
                self.all_proposal_images.extend(proposal_images)
                self.all_proposal_targets.extend(proposal_targets)
                
                count += 1
                print(count)
                if count > 2:
                    break

            pickle_save(self.all_proposal_images, self.all_proposal_targets, train='train')
            #self.final_image, self.final_target = class_imbalance(all_proposal_images, all_proposal_target)

            #save(self.final_image, self.final_target)





    def __len__(self):
        """Returns the total number of samples in the dataset."""
        return len(self.all_proposal_images)

    def __getitem__(self, idx):
        #return self.final_image[idx], self.final_target[idx]
        return self.all_proposal_images[idx], self.all_proposal_targets[idx]



def pickle_save(final_image, final_target, train):
    if train:
        folder_path = os.path.join('..', 'Potholes', 'train_proposals')

        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        with open(os.path.join(folder_path, 'train_image.pkl'), 'wb') as f:
            pk.dump(final_image, f)
        with open(os.path.join(folder_path, 'train_target.pkl'), 'wb') as f:
            pk.dump(final_target, f)
#    if val:

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

    dataloader = DataLoader(prop, batch_size=32, shuffle=True, num_workers=8, collate_fn=custom_collate_fn)
    #visualize_samples(dataloader, num_images=4, figname='pothole_samples', box_thickness=5)



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