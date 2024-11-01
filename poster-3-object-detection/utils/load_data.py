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
from torch.utils.data import default_collate


class Potholes(Dataset):
    """
    A PyTorch Dataset class for loading images and annotations of potholes.

    Attributes:
        folder_path (str): Path to the dataset folder with the splits.json file, annotated-images folder and README.md.
        transform (callable, optional): Optional transform to be applied on a sample.
        image_paths (list): List of file paths for images.
        xml_paths (list): List of file paths for corresponding XML annotation files.

    Parameters:
        split (str): The dataset split to use ('train', 'val', or 'test'). Defaults to 'train'.
        val_percent (int, optional): The proportion of training data to use for validation.
                                        If provided, the dataset will split the training data.
        seed (int): Seed for random shuffling of the training data. Defaults to 42.
        transform (callable, optional): A function/transform to apply to the images.

    Raises:
        FileNotFoundError: If the specified folder path does not exist.
        AssertionError: If the number of image paths does not match the number of XML paths.

    Methods:
        __len__(): Returns the total number of samples in the dataset.
        __getitem__(idx): Retrieves a sample (image and targets) from the dataset at the given index.
    """

    def __init__(self, split='train', val_percent=None, seed=42, transform=None, folder_path='Potholes'):
        # Ensure the dataset is accessed from the root of the repository
        base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        self.folder_path = os.path.join(base_path, folder_path)
        self.transform = transform

        # Check if the folder path exists
        if not os.path.exists(self.folder_path):
            print("Looking for files in:", self.folder_path)
            raise FileNotFoundError(f"Directory not found: {self.folder_path}")

        # Load the splits from the JSON file
        json_path = os.path.join(self.folder_path, "splits.json")
        with open(json_path, 'r') as file:
            splits = json.load(file)

        #If the validation percentage for the split is set, it will create a validation set based on the existing training set
        if val_percent is not None:
            #Get all the files to calculate the precentage for validation set
            number_of_all_files = len(sorted(glob.glob(os.path.join(self.folder_path, "annotated-images/img-*.jpg")))) #Get the number of all the files in the folder 
            train_files = splits['train']
            random.seed(seed)
            random.shuffle(train_files)  

            # Calculate the number of validation samples
            val_count = int(number_of_all_files * val_percent/100)
            new_val_files = train_files[:val_count]
            new_train_files = train_files[val_count:]

            # Set the paths based on the wanted splits
            # In the code below we use the replace to replace '.xml' with '.jpg' because the files in the given json only consist of '.xml' files.
            # This is used to get the path for both the '.xml' and '.jpg' files
            if split.lower() == 'train':
                self.image_paths = [os.path.join(self.folder_path, "annotated-images", file.replace('.xml', '.jpg')) for file in new_train_files]
                self.xml_paths = [os.path.join(self.folder_path, "annotated-images", file) for file in new_train_files]
            elif split.lower() == 'val':
                self.image_paths = [os.path.join(self.folder_path, "annotated-images", file.replace('.xml', '.jpg')) for file in new_val_files]
                self.xml_paths = [os.path.join(self.folder_path, "annotated-images", file) for file in new_val_files]
            elif split.lower() == 'test':
                self.image_paths = [os.path.join(self.folder_path, "annotated-images", file.replace('.xml', '.jpg')) for file in splits['test']]
                self.xml_paths = [os.path.join(self.folder_path, "annotated-images", file) for file in splits['test']]
        else:
            # Use the original splits if val_percent is not provided
            if split.lower() == 'train':
                self.image_paths = [os.path.join(self.folder_path, "annotated-images", file.replace('.xml', '.jpg')) for file in splits['train']]
                self.xml_paths = [os.path.join(self.folder_path, "annotated-images", file) for file in splits['train']]
            elif split.lower() == 'test':
                self.image_paths = [os.path.join(self.folder_path, "annotated-images", file.replace('.xml', '.jpg')) for file in splits['test']]
                self.xml_paths = [os.path.join(self.folder_path, "annotated-images", file) for file in splits['test']]

        assert len(self.image_paths) == len(self.xml_paths), 'Number of images and xml files does not match'


    def __len__(self):
        """Returns the total number of samples in the dataset."""
        return len(self.image_paths)

    def __getitem__(self, idx):
        """Retrieves a sample (image and targets) from the dataset at the given index.

        Args:
            idx (int): The index of the sample to retrieve.

        Returns:
            tuple: A tuple containing:
                - image (Tensor): The image tensor after applying the transformations.
                - targets (list): A list of TensorDict objects containing bounding box coordinates and labels.
        """

        # Load the image and convert to RGB
        image = Image.open(self.image_paths[idx]).convert('RGB')
        original_width, original_height = image.size

        # Load and parse the XML annotation
        xml_path = self.xml_paths[idx]
        tree = ET.parse(xml_path)
        root = tree.getroot()

        # Initialize lists for the target
        targets = []

        # The code below convert the image to a tensor
        # If a transform is set, then it will calculate 
        if self.transform:
            image = self.transform(image)
            new_height, new_width = image.shape[1], image.shape[2]
        else:
            image = transform.ToTensor(image)

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

            # Apply transformations and reshape so the boxes match to the new size
            if self.transform:
                xmin *= new_width / original_width
                xmax *= new_width / original_width
                ymin *= new_height / original_height
                ymax *= new_height / original_height


            # Append bounding box and label. TensorDict is used to convert the dictorary to a tensor
            directory = TensorDict({
                'xmin'  : torch.tensor(xmin, dtype=torch.float32, device=device),
                'ymin'  : torch.tensor(ymin, dtype=torch.float32, device=device),
                'xmax'  : torch.tensor(xmax, dtype=torch.float32, device=device),
                'ymax'  : torch.tensor(ymax, dtype=torch.float32, device=device),
                'labels': torch.tensor(label, dtype=torch.int64, device=device)
            })

            targets.append(directory)
        return image, targets


def load_data(val_percent=None, seed=42, transform=None, folder_path='Potholes'):
    """
    Loads the Potholes dataset for training, validation, and testing.

    Parameters:
        val_percent (int, optional): The proportion of the training data to use for validation.
                                        If provided, the training set will be split accordingly.
        seed (int): Seed for random shuffling of the training data. Defaults to 42.
        transform (callable, optional): A function/transform to apply to the images.
        folder_path (str): Relative path to the folder containing the dataset. Defaults to 'Potholes'.

    Returns:
        tuple: A tuple containing three elements:
            - train_data (Potholes): The dataset for training.
            - val_data (Potholes): The dataset for validation.
            - test_data (Potholes): The dataset for testing.
    """
    train_data = Potholes(split='train', val_percent=val_percent, seed=seed, transform=transform, folder_path=folder_path)
    val_data = Potholes(split='val', val_percent=val_percent, seed=seed, transform=transform, folder_path=folder_path)
    test_data = Potholes(split='test', val_percent=val_percent, seed=seed, transform=transform, folder_path=folder_path)

    return train_data, val_data, test_data

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

    # Define any transforms
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])
    device = 'cpu'
    # Initialize the dataset and dataloader
    potholes_dataset = Potholes(split = 'Train', transform=transform, folder_path='Potholes')
    dataloader = DataLoader(potholes_dataset, batch_size=32, shuffle=True, num_workers=8, collate_fn=custom_collate_fn)

    print("\nNumber of samples in the dataset:", len(potholes_dataset))
    print("Number of batches in the dataloader:", len(dataloader))

    #Check the get item method
    sample_image, sample_targets = potholes_dataset[0]  
    print("\nSample Image Type:", type(sample_image))
    print("Image in on the following device (-1 = cpu) and (0 = cuda):", sample_image.get_device())
    print("Sample Targets Type:", type(sample_targets))
    
    # Check the type of individual targets
    target = sample_targets[0]
    print("\nType of individual target:", type(target))
    print("Type of xmin:", type(target['xmin']))
    print("Type of labels:", type(target['labels']))
  
    #Check the dataloader
    data_iter = iter(dataloader)
    batch_images, batch_targets = next(data_iter)

    #When all the data is loaded we can insert it to the GPU if it is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('You are using:', device)
    batch_images = batch_images[0].to(device)        #Check to ensure the data can be send to the cuda:0 
    targets = batch_targets[0]
    box = targets[0].to(device)

    print("\nSingle Batch:")
    print("The batch is on the")
    print("Image batch in on the following device (-1 = cpu) and (0 = cuda):", len(batch_targets)) 
    print("Image batch shape:", batch_images.shape) 

    # Print the target from the dataloader
    # Batch_target is all the targets in the batch whereas targets is for 1 image (the boxes on the image). Box is therefore one of the boxes on the image (targets)
    print(f'\nPrint the box:\n {box}')
    print("Type:", type(box))

    print("\nBounding box coordinates:", box['xmin'], box['ymin'], box['xmax'], box['ymax'])
    print("Label:", box['labels'])

    #The following code is used to show that the output from costum_collate_fn works
    device = 'cpu'
    for batch_images, batch_targets in dataloader:
        print("Batch images shape:", batch_images.shape)  # Should print: (2, 3, 256, 256)
        print("\nTargets for each image in the batch:")

        # Iterate through the targets to show correspondence with images
        for i, targets in enumerate(batch_targets):
            print(f"Image {i} has {len(targets)} target(s):")
            for target in targets:
                print(f"  Bounding box: {target['xmin']:.3g}, {target['ymin']:.3g}, {target['xmax']:.3g}, {target['ymax']:.3g}, Label: {target['labels']}")
            if i >= 5:
                break # only show 5 images
        break         # Only show one batch

    #The following code is used to check that the precentage for train, valdiation and test is working
    train_dataset, val_dataset, test_dataset = load_data(val_percent=20, seed=42, transform=transform)
    total = len(train_dataset) + len(val_dataset) + len(test_dataset)
    print('The split for train is:', len(train_dataset)/total)
    print('The split for validation is:', len(val_dataset)/total)
    print('The split for test is:', len(test_dataset)/total)



###############################################################
    #Function to benchmark the dataloader

    #def benchmark_dataloader(dataloader, num_batches=100):
    #    start_time = time.time()
    #    for i, (images, targets) in enumerate(dataloader):
    #        if i >= num_batches:
    #            break
    #    end_time = time.time()
    #    return end_time - start_time
    ##Test for optimal num of workers in DataLoader
    #
    ## BEST IN NUM_WORKERS = 4
    #batch_size = 32
    #num_workers_list = [0, 2, 4, 8, 16, 32, 64]
    #
    #for num_workers in num_workers_list:
    #    dataloader = DataLoader(
    #        potholes_dataset,
    #        batch_size=batch_size,
    #        shuffle=True,
    #        num_workers=num_workers,
    #        collate_fn=collate_fn,
    #    )
    #    duration = benchmark_dataloader(dataloader)
    #    print(f"num_workers: {num_workers}, Time taken: {duration:.2f} seconds")
    #
    #benchmark_dataloader(dataloader, num_batches=64)
    