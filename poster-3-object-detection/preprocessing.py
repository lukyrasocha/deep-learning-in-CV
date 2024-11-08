import os
import json
import random
import glob

from PIL import Image
from utils.load_data import get_xml_data, pickle_save, class_balance
from utils.selective_search import generate_proposals_and_targets
from torchvision import transforms
        

if __name__ == '__main__':

    #The paths is relative to being in the poster-3-object-detection folder
    get_images_from_folder_relative = 'Potholes/annotated-images'
    get_split_from_folder_relative = 'Potholes'
    save_data_in_folder_relative = 'Potholes/training_data'
    seed = 42
    val_percent = 20
    iou_upper_limit = 0.5
    iou_lower_limit = 0.5
    method = 'quality'
    max_proposals = 2000
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        ])

    # Ensure the dataset is accessed from the root of the repository
    base_path = os.path.abspath(os.path.join(os.path.dirname(__file__)))
    get_images_from_folder_full = os.path.join(base_path, get_images_from_folder_relative)
    save_data_in_folder_full = os.path.join(base_path, save_data_in_folder_relative)

    # Check if the folder paths exists
    if not os.path.exists(get_images_from_folder_full):
        raise SystemExit(f"Error: Directory not found: {get_images_from_folder_full}")
    elif not os.path.exists(save_data_in_folder_full):
        raise SystemExit(f"Error: Directory not found: {save_data_in_folder_full}")


    # Load the splits from the JSON file
    json_path = os.path.join(get_split_from_folder_relative, "splits.json")
    with open(json_path, 'r') as file:
        splits = json.load(file)
    train_files = splits['train']

    #If the validation percentage for the split is set, it will create a validation set based on the existing training set
    if val_percent is not None:
        random.seed(seed)
        random.shuffle(train_files) 
        #Get all the files to calculate the precentage for validation set
        number_of_all_files = len(sorted(glob.glob(os.path.join(get_images_from_folder_full, 'img-*.jpg')))) #Get the number of all the files in the folder 

        # Calculate the number of validation samples
        val_count = int(number_of_all_files * val_percent/100)
        new_val_files = train_files[:val_count]
        new_train_files = train_files[val_count:]

        image_paths = [os.path.join(get_images_from_folder_full, file.replace('.xml', '.jpg')) for file in new_train_files]
        xml_paths = [os.path.join(get_images_from_folder_full, file) for file in new_train_files]
    else:
        image_paths = [os.path.join(get_images_from_folder_full, file.replace('.xml', '.jpg')) for file in train_files]
        xml_paths = [os.path.join(get_images_from_folder_full, file) for file in train_files]

    count = 0
    for image_path, xml_path in zip(image_paths, xml_paths):

        original_image = Image.open(image_path).convert('RGB')
        original_targets = get_xml_data(xml_path)
        # Get the image id for pickling ids
        image_id = image_path.split('/')[-1].split('.')[0]
        
        # print out every 50th image id for tracking while running
        if count % 50 == 0:
            print(f"Image id {count}")

        # Generate proposals and targets
        proposal_images, proposal_targets = generate_proposals_and_targets(original_image, original_targets, transform, image_id, iou_upper_limit, iou_lower_limit, method, max_proposals, generate_target = True)
        proposal_images_balanced, proposal_targets_balanced = class_balance(proposal_images, proposal_targets, seed, count)

        if proposal_images_balanced is not None:
            pickle_save(proposal_images_balanced, proposal_targets_balanced, save_data_in_folder_full, train=True, index=image_id)
        count += 1

        break
