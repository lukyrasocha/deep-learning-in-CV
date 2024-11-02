import os
import glob
import time
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import xml.etree.ElementTree as ET
from torchvision import transforms
from visualize import visualize_samples
from tensordict import TensorDict
import json
import cv2 as cv
import argparse 
import random



class Potholes(Dataset):
    def __init__(self, split='train', val_percentage=None, seed=42, transform=None, folder_path='Potholes'):
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
        if val_percentage is not None:

            #Get all the training files and shuffel them
            train_files = splits['train']
            random.seed(seed)
            random.shuffle(train_files)  

            # Calculate the number of validation samples
            val_count = int(len(train_files) * val_percentage)
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
            # Use the original splits if val_percentage is not provided
            if split.lower() == 'train':
                self.image_paths = [os.path.join(self.folder_path, "annotated-images", file.replace('.xml', '.jpg')) for file in splits['train']]
                self.xml_paths = [os.path.join(self.folder_path, "annotated-images", file) for file in splits['train']]
            elif split.lower() == 'test':
                self.image_paths = [os.path.join(self.folder_path, "annotated-images", file.replace('.xml', '.jpg')) for file in splits['test']]
                self.xml_paths = [os.path.join(self.folder_path, "annotated-images", file) for file in splits['test']]

        assert len(self.image_paths) == len(self.xml_paths), 'Number of images and xml files does not match'


    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
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
            directory = TensorDict({'xmin' : xmin, 'ymin': ymin, 'xmax': xmax, 'ymax': ymax, 'labels': label})
            targets.append(directory)


        return image, targets

def load_data(split, transform, folder_path='Potholes', seed=42):
    None





if __name__ == "__main__":
    # Define any transforms
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])

    # Initialize the dataset and dataloader
    potholes_dataset = Potholes(split = 'Train', transform=transform, folder_path='Potholes')
    dataloader = DataLoader(potholes_dataset, batch_size=32, shuffle=True, num_workers=8)

    print("Number of samples in the dataset:", len(potholes_dataset))
    print("Number of batches in the dataloader:", len(dataloader))


    sample_image, sample_targets = potholes_dataset[0]  
    print("Sample Image Type:", type(sample_image))
    print("Sample Targets Type:", type(sample_targets))
    
    # Check the type of individual targets
    target = sample_targets[0]
    print("Type of individual target:", type(target))
    print("Type of xmin:", type(target['xmin']))
    print("Type of labels:", type(target['labels']))
    #Visualize samples
    #visualize_samples(dataloader, num_images=4, figname='pothole_samples', box_thickness=5)

    # get image from the dataloader 
    image, targets = potholes_dataset[100]
    

    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", required=True,
        help="path to the input image")
    ap.add_argument("-m", "--method", type=str, default="fast",
        choices=["fast", "quality"],
        help="selective search method")
    args = vars(ap.parse_args())

    image = cv.imread(args["image"])
    ss = cv.ximgproc.segmentation.createSelectiveSearchSegmentation()
    ss.setBaseImage(image)

    # check to see if we are using the *fast* but *less accurate* version
    # of selective search
    if args["method"] == "fast":
        print("[INFO] using *fast* selective search")
        ss.switchToSelectiveSearchFast()
    # otherwise we are using the *slower* but *more accurate* version
    else:
        print("[INFO] using *quality* selective search")
        ss.switchToSelectiveSearchQuality()


        # run selective search on the input image
    start = time.time()
    rects = ss.process()
    end = time.time()
    # show how along selective search took to run along with the total
    # number of returned region proposals
    print("[INFO] selective search took {:.4f} seconds".format(end - start))
    print("[INFO] {} total region proposals".format(len(rects)))






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
    ##BEST IN NUM_WORKERS = 8
    #batch_size = 32
    #num_workers_list = [0, 2, 4, 8, 16, 32, 64]
    #for num_workers in num_workers_list:
    #    dataloader = DataLoader(
    #        potholes_dataset,
    #        batch_size=batch_size,
    #        shuffle=True,
    #        num_workers=num_workers,
    #    )
    #    duration = benchmark_dataloader(dataloader)
    #    print(f"num_workers: {num_workers}, Time taken: {duration:.2f} seconds")
    #
    #
    #benchmark_dataloader(dataloader, num_batches=64)
    #