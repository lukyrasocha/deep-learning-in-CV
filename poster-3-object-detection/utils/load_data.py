import glob
import torch
import os
import random

from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
import numpy as np
import json


import os
import xml.etree.ElementTree as ET

def parse_annotation(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()

    # Get the image filename
    filename = root.find('filename').text

    # Initialize list to hold all objects in the image
    objects = []

    # Iterate over all object elements in the XML
    for obj in root.findall('object'):
        category = obj.find('name').text
        
        # Get bounding box coordinates
        bndbox = obj.find('bndbox')
        x_min = int(bndbox.find('xmin').text)
        y_min = int(bndbox.find('ymin').text)
        x_max = int(bndbox.find('xmax').text)
        y_max = int(bndbox.find('ymax').text)
        
        # Add the object info to the list
        objects.append({
            "category": category,
            "x_min": x_min,
            "x_max": x_max,
            "y_min": y_min,
            "y_max": y_max
        })

    # Return the image info and objects list
    return {
        "image_name": filename,
        "objects": objects
    }

def parse_annotations_from_folder(folder_path):
    data = []
    
    # Loop through all files in the folder
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.xml'):
            # Parse the XML file
            xml_file = os.path.join(folder_path, file_name)
            image_data = parse_annotation(xml_file)
            data.append(image_data)
    
    return data

#print(os.path.dirname(os.path.abspath(__file__)))
path, _ = os.path.split(os.path.dirname(os.path.abspath(__file__)))

folder_path = os.path.join(path, "Potholes/annotated-images")


data = parse_annotations_from_folder(folder_path)


# Print the result
#for image_data in data:
#    print(image_data)


#with open('data.json', 'w') as f:
#    json.dump(data, f)

