import os
import glob
import time
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import xml.etree.ElementTree as ET
from torchvision import transforms
from visualize import visualize_samples
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

        print("Looking for files in:", self.folder_path)
        if not os.path.exists(self.folder_path):
            raise FileNotFoundError(f"Directory not found: {self.folder_path}")

        self.image_paths = sorted(glob.glob(os.path.join(self.folder_path, "img-*.jpg")))
        self.xml_paths = sorted(glob.glob(os.path.join(self.folder_path, "img-*.xml")))
        print(f"Found {len(self.image_paths)} images and {len(self.xml_paths)} XML files.")
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

        # Initialize lists for bounding boxes and labels
        boxes = []
        labels = []

        # Iterate through each object in the XML file
        for obj in root.findall('object'):
            label = obj.find('name').text
            labels.append(label)

            # Extract bounding box coordinates
            bndbox = obj.find('bndbox')
            xmin = int(bndbox.find('xmin').text)
            ymin = int(bndbox.find('ymin').text)
            xmax = int(bndbox.find('xmax').text)
            ymax = int(bndbox.find('ymax').text)

            # Append bounding box as [xmin, ymin, xmax, ymax]
            boxes.append([xmin, ymin, xmax, ymax])

        # Convert to tensors
        boxes = torch.tensor(boxes, dtype=torch.float32)
        labels = torch.tensor([1 if label == 'pothole' else 0 for label in labels], dtype=torch.int64)

        # Apply transformations
        if self.transform:
            image = self.transform(image)
            new_height, new_width = image.shape[1], image.shape[2]

            # Scale boxes
            scale_x = new_width / original_width
            scale_y = new_height / original_height
            boxes[:, [0, 2]] = boxes[:, [0, 2]] * scale_x
            boxes[:, [1, 3]] = boxes[:, [1, 3]] * scale_y

        target = {'boxes': boxes, 'labels': labels}
        #print(f"Loaded image {idx} with {len(boxes)} bounding boxes")

        return image, target

def collate_fn(batch):
    images = []
    targets = []
    for image, target in batch:
        images.append(image)
        targets.append(target)
    images = torch.stack(images, dim=0)
    return images, targets


# Function to benchmark the dataloader

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

