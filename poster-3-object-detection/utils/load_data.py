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
# balba

class Potholes(Dataset):
    def __init__(self, transform=None, folder_path='Potholes/annotated-images'):
        # This ensures that we can access the dataset from the root of the repository
        # Since we do not have the dataset in the same directory as provided by the 
        # teachers to us in a previous assignment and the root folder would be 
        # different to all of us. This takes care of that. 
        base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        self.folder_path = os.path.join(base_path, folder_path)
        self.transform = transform


        if not os.path.exists(self.folder_path):
            print("Looking for files in:", self.folder_path)
            raise FileNotFoundError(f"Directory not found: {self.folder_path}")

        self.image_paths = sorted(glob.glob(os.path.join(self.folder_path, "img-*.jpg")))
        self.xml_paths = sorted(glob.glob(os.path.join(self.folder_path, "img-*.xml")))
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
                image = self.transform(image)
                new_height, new_width = image.shape[1], image.shape[2]

                xmin *= new_width / original_width
                xmax *= new_width / original_width
                ymin *= new_height / original_height
                ymax *= new_height / original_height

            # Append bounding box and label. TensorDict is used to convert the dictorary to a tensor
            directory = TensorDict({'xmin' : xmin, 'ymin': ymin, 'xmax': xmax, 'ymax': ymax, 'labels': label})
            targets.append(directory)


        return image, targets

def collate_fn(batch):
    images = []
    targets = []
    for image, target in batch:
        images.append(image)
        targets.append(target)
    images = torch.stack(images, dim=0)
    return images, targets

def load_data(split, transform, folder_path='Potholes/annotated-images', seed=42):
    None

 #Function to benchmark the dataloader

#def benchmark_dataloader(dataloader, num_batches=100):
#    start_time = time.time()
#    for i, (images, targets) in enumerate(dataloader):
#        if i >= num_batches:
#            break
#    end_time = time.time()
#    return end_time - start_time



if __name__ == "__main__":
    # Define any transforms
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])

    # Initialize the dataset and dataloader
    potholes_dataset = Potholes(transform=transform, folder_path='Potholes/annotated-images')
    dataloader = DataLoader(potholes_dataset, batch_size=32, shuffle=True, num_workers=8, collate_fn=collate_fn)

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

    # Visualize samples
    visualize_samples(dataloader, num_images=4, figname='pothole_samples', box_thickness=5)
    



###############################################################
# Test for optimal num of workers in DataLoader

# BEST IN NUM_WORKERS = 8

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
#
    #benchmark_dataloader(dataloader, num_batches=64)

