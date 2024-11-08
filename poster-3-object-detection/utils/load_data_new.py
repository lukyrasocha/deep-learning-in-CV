import os
import glob
import time
import torch
import json
import xml.etree.ElementTree as ET
import random
import pickle as pk
import json

from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tensordict import TensorDict
from visualize import visualize_samples, visualize_proposal
from torch.utils.data import default_collate
from selective_search_new import  generate_proposals_and_targets
from torch.utils.data import default_collate


class Training_data(Dataset):
    
    def __init__(self, val_percent=None, seed=42, transform=None, folder_path='Potholes', iou_upper_limit=0.5, iou_lower_limit=0.5, method='fast', max_proposals=2000, training_file_name = None):

        assert training_file_name is not None, 'Specify a filename to open or to create'

        try:
            # Attempt to load the data
            print('it will open')
            self.proposal_image_dataset, self.proposal_target_dataset = pickle_load(training_file_name, train=True)
            
        except Exception as e:
            print('It will generate')

            #The to lists contains a number of proposals and targets that is going to be used in training 
            all_proposal_images = []
            all_proposal_targets = []
        
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

            train_files = splits['train']
            random.seed(seed)
            random.shuffle(train_files)  

            #If the validation percentage for the split is set, it will create a validation set based on the existing training set
            if val_percent is not None:
                #Get all the files to calculate the precentage for validation set
                number_of_all_files = len(sorted(glob.glob(os.path.join(entire_folder_path, "annotated-images/img-*.jpg")))) #Get the number of all the files in the folder 

                # Calculate the number of validation samples
                val_count = int(number_of_all_files * val_percent/100)
                new_val_files = train_files[:val_count]
                new_train_files = train_files[val_count:]

                image_paths = [os.path.join(entire_folder_path, "annotated-images", file.replace('.xml', '.jpg')) for file in new_train_files]
                xml_paths = [os.path.join(entire_folder_path, "annotated-images", file) for file in new_train_files]
            else:
                image_paths = [os.path.join(entire_folder_path, "annotated-images", file.replace('.xml', '.jpg')) for file in train_files]
                xml_paths = [os.path.join(entire_folder_path, "annotated-images", file) for file in train_files]

            for image_path, xml_path in zip(image_paths, xml_paths):
                original_image_name = os.path.splitext(os.path.basename(image_path))[0]
                original_image = Image.open(image_path).convert('RGB')
                original_targets = get_xml_data(xml_path)

                proposal_images, proposal_targets = generate_proposals_and_targets(original_image, original_targets, transform, original_image_name, iou_upper_limit, iou_lower_limit, method, max_proposals,  generate_target=True)
                all_proposal_images.extend(proposal_images)
                all_proposal_targets.extend(proposal_targets)
                break

            self.proposal_image_dataset = all_proposal_images
            self.proposal_target_dataset = all_proposal_targets

            #pickle_save(self.proposal_image_dataset, self.proposal_target_dataset, True, training_file_name)
            #save_tensors_with_torch(self.proposal_image_dataset, self.proposal_target_dataset, training_file_name)

    def __len__(self):
        """Returns the total number of samples in the dataset."""
        return len(self.proposal_image_dataset)

    def __getitem__(self, idx):
        return self.proposal_image_dataset[idx], self.proposal_target_dataset[idx]


class Val_and_test_data(Dataset):
    def __init__(self, split='val', val_percent=None, seed=42, transform=None, folder_path='Potholes', iou_upper_limit=0.5, iou_lower_limit=0.5, method='fast', max_proposals=int(2000)):

        self.all_proposal_images = []
        self.all_proposal_targets = []
        self.all_original_images = []
        self.all_original_targets = []

        # Ensure the dataset is accessed from the root of the repository
        base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        entire_folder_path = os.path.join(base_path, folder_path)

        # Load the splits from the JSON file
        json_path = os.path.join(entire_folder_path, "splits.json")
        with open(json_path, 'r') as file:
            splits = json.load(file)

        # Check if the folder path exists
        if not os.path.exists(entire_folder_path):
            raise FileNotFoundError(f"Directory not found: {entire_folder_path}")

            train_files = splits['train']
            random.seed(seed)
            random.shuffle(train_files)  

        if split.lower() == 'val':

            #If the validation percentage for the split is set, it will create a validation set based on the existing training set
            if val_percent is not None:
                #Get all the files to calculate the precentage for validation set
                number_of_all_files = len(sorted(glob.glob(os.path.join(entire_folder_path, "annotated-images/img-*.jpg")))) #Get the number of all the files in the folder 

                # Calculate the number of validation samples
                val_count = int(number_of_all_files * val_percent/100)
                new_val_files = train_files[:val_count]

                image_paths = [os.path.join(entire_folder_path, "annotated-images", file.replace('.xml', '.jpg')) for file in new_val_files]
                xml_paths = [os.path.join(entire_folder_path, "annotated-images", file) for file in new_val_files]
            else:
                raise ValueError('Validation precentage is not set')

        elif split.lower() == 'test':
                image_paths = [os.path.join(entire_folder_path, "annotated-images", file.replace('.xml', '.jpg')) for file in splits['test']]
                xml_paths = [os.path.join(entire_folder_path, "annotated-images", file) for file in splits['test']]
        else:
            raise ValueError('Use either val or test to access data')

        assert len(image_paths) == len(xml_paths), "The length of images and xml files is not the same"

        count = 0
        for image_path, xml_path in zip(image_paths, xml_paths):
            original_image_name = os.path.splitext(os.path.basename(image_path))[0]
            original_image = Image.open(image_path).convert('RGB')
            original_targets = get_xml_data(xml_path)

            proposal_images, proposal_targets = generate_proposals_and_targets(original_image, original_targets, transform, original_image_name, iou_upper_limit, iou_lower_limit, method, max_proposals, generate_target = False)
            self.all_proposal_images.append(proposal_images)
            self.all_proposal_targets.append(proposal_targets)
            self.all_original_images.append(transform(original_image))
            self.all_original_targets.append(original_targets)
            count += 1
            print(f"Processed {count} images.")
            if count > 8:
                break

    def __len__(self):
        return len(self.all_original_images)


    def __getitem__(self, idx):
        return self.all_original_images[idx], self.all_original_targets[idx], self.all_proposal_images[idx], self.all_proposal_targets[idx]




def pickle_save(image_to_save, target_to_save, train, file_name):
    if train:
        folder_path = os.path.join('..', 'Potholes', 'train_proposals')

        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        with open(os.path.join(folder_path, file_name + '_image.pkl'), 'wb') as f:
            pk.dump(image_to_save, f)
        with open(os.path.join(folder_path, file_name + '_target.pkl'), 'wb') as f:
            pk.dump(target_to_save, f)

def pickle_load(file_name, train):

    if train:
        folder_path = os.path.join('..', 'Potholes', 'train_proposals')

        image_file = os.path.join(folder_path, file_name + '_image.pkl')
        target_file = os.path.join(folder_path, file_name + '_target.pkl')

        # Load the image and target from the pickle files
        with open(image_file, 'rb') as f:
            image = pk.load(f)
        
        with open(target_file, 'rb') as f:
            target = pk.load(f)
        
        return image, target
    else:
        raise ValueError('The function currently supports loading from the training folder only.')

def save_tensors_with_torch(list1, list2, filename):

    folder_path = os.path.join('..', 'Potholes', 'train_proposals')

    torch.save({'list1': list1, 'list2': list2}, os.path.join(folder_path, filename))

def load_tensors_with_torch(filename):
    data = torch.load(filename)
    return data['list1'], data['list2']

    # Loading
    list1, list2 = load_tensors_with_torch('tensors.pth')


def training_collate_fn(batch):
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




if __name__ == "__main__":
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])
    #prop = Training_data(val_percent=20, transform=transform, folder_path='Potholes', training_file_name = 'Trainingset3')
    #dataloader = DataLoader(prop, batch_size=32, shuffle=True, num_workers=8, collate_fn=training_collate_fn)
    #count = 0
    #for batch_images, batch_targets in dataloader:
    #    for image, target in zip(batch_images, batch_targets):
    #        print(image.shape, target)
    #        print()
    #        count += 1
    #        if count > 10:
    #            break
    start_time = time.time()
    val = Val_and_test_data(split='test', val_percent=20, transform=transform, folder_path='Potholes')
    dataloader_val = DataLoader(val, batch_size = 32, shuffle=True, num_workers=8, collate_fn=val_test_collate_fn)
    end_time = time.time()
    print("Time taken to load one batch:", end_time - start_time, "seconds")


    count = 0
    for batch_original_images, batch_original_targets, batch_proposal_images, batch_proposal_targets in dataloader_val:
    
        print("Batch original images shape:", batch_original_images.shape)
        print("Batch original targets:", batch_original_targets)
        print("Number of batch proposal images:", len(batch_proposal_images))
        print("Shape of first batch proposal image:", batch_proposal_images[0].shape)
        print("Number of batch proposal targets:", len(batch_proposal_targets))
        print("First batch proposal target:", batch_proposal_targets[0])


    #visualize_samples(dataloader, num_images=4, figname='pothole_samples', box_thickness=5)
















# image_list, taget_list (not in balance)
# image_balance, target_balnce = class()
#
#count_idx = 0
# loop through image_balance
# save image_balance, target_balance with idx
# count_idx += 1