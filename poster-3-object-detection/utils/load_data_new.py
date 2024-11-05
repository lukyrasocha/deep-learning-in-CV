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
from visualize import visualize_samples
import json
from torch.utils.data import default_collate
from selective_search_new import  get_proposals_and_targets




class Proposals(Dataset):

    def __init__(self, split='train', val_percent=None, seed=42, transform=None, folder_path='Potholes'):

        #The to lists contains a number of proposals and targets that is going to be used in training 
        self.image_proposals_list = []
        self.target_proposals_list = []
    
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

            for image_path, xml_path in zip(image_paths, xml_paths):
                image = Image.open(image_path).convert('RGB')
                image_target = get_xml_data(xml_path)

                proposals, proposal_targets = get_proposals_and_targets(image, image_target, transform)

                print('1 iteration')







            ############################################################
            #               Create the following:
            #
            # tree = ET.parse(xml_path)
            # root = tree.getroot()
            #
            # for idx in range(self.image_paths)  
            #    Get image and target dictonary
            #    
            #    proposal, taget = get_proposal  
            #
            #    proposals.extend(proposal)
            #    targets.extend(target)
            #    
            #    
            #    
            #

            


    def __len__(self):
        """Returns the total number of samples in the dataset."""
        return len(self.image_paths)

    def __getitem__(self, idx):
        None

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




if __name__ == "__main__":
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])
    prop = Proposals(split = 'Train', val_percent=20, transform=transform, folder_path='Potholes')

    print('hello')