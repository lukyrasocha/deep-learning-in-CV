import os
import json
import pickle
import random
import glob
import sys

from PIL import Image
from utils.load_data import get_xml_data, pickle_save, class_balance, save_ground_truth
from utils.selective_search import generate_proposals_and_targets_for_training, generate_proposals_for_test_and_val
from torchvision import transforms
from utils.logger import logger

def ensure_dir(directory):
    if not os.path.exists(directory):
        print(f"Directiory {(directory)} does not exists, Creating directory: ", directory)
        os.makedirs(directory)

if __name__ == '__main__':

    TRAIN_PROPOSALS = True
    VALIDATION_PROPOSALS = True
    TEST_PROPOSALS = True 

    blackhole_path = os.getenv('BLACKHOLE')
    if not blackhole_path:
        raise EnvironmentError("The $BLACKHOLE environment variable is not set or is empty.")


    #The paths is relative to being in the poster-3-object-detection folder
    get_images_from_folder_relative = 'Potholes/annotated-images'
    get_split_from_folder_relative = 'Potholes'

    # save paths for training 
    save_images_in_folder_relative = os.path.join(blackhole_path, 'DLCV/training_data/images')
    save_targets_in_folder_relative = os.path.join(blackhole_path, 'DLCV/training_data/targets')

    # save paths for validation
    save_targets_in_folder_relative_val = os.path.join(blackhole_path, 'DLCV/val_data/targets') #'Potholes/validation_data/targets'

    # save paths for test
    save_targets_in_folder_relative_test = os.path.join(blackhole_path, 'DLCV/test_data/targets') # 'Potholes/test_data/targets'



    SEED = 42
    VAL_PERCENT = 20 
    IOU_UPPER_LIMIT = 0.5
    IOU_LOWER_LIMIT = 0.5
    METHOD = 'quality'
    MAX_PROPOSALS = 500

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        ])

    # Ensure the dataset is accessed from the root of the repository
    base_path = os.path.abspath(os.path.join(os.path.dirname(__file__)))
    get_images_from_folder_full = os.path.join(base_path, get_images_from_folder_relative)
    save_images_in_folder_full = os.path.join(base_path, save_images_in_folder_relative)
    save_targets_in_folder_full = os.path.join(base_path, save_targets_in_folder_relative)
    save_targets_in_folder_full_val = os.path.join(base_path, save_targets_in_folder_relative_val)
    save_targets_in_folder_full_test = os.path.join(base_path, save_targets_in_folder_relative_test)

    # Ensure the directories exist
    ensure_dir(get_images_from_folder_full)
    ensure_dir(save_images_in_folder_full)
    ensure_dir(save_targets_in_folder_full)
    ensure_dir(save_targets_in_folder_full_val)
    ensure_dir(save_targets_in_folder_full_test)
    


#    # Load the splits from the JSON file
    json_path = os.path.join(get_split_from_folder_relative, "splits.json")
    with open(json_path, 'r') as file:
        splits = json.load(file)
    train_files = splits['train']
    test_files = splits['test']

#    #If the validation percentage for the split is set, it will create a validation set based on the existing training set
    if VAL_PERCENT is not None:
        random.seed(SEED)
        random.shuffle(train_files) 
        
                #Get all the files to calculate the precentage for validation set
                #number_of_all_files = len(sorted(glob.glob(os.path.join(get_images_from_folder_full, 'img-*.jpg')))) #Get the number of all the files in the folder 

        # Calculate the number of validation samples
        val_count = int(len(train_files) * VAL_PERCENT/100)
        new_val_files = train_files[:val_count]
        new_train_files = train_files[val_count:]
    else:
        raise Exception("Validation percentage is not set")
    
    if TRAIN_PROPOSALS:
        logger.working_on(f"Creating training proposals with and targets {len(new_train_files)} images")
        image_paths = [os.path.join(get_images_from_folder_full, file.replace('.xml', '.jpg')) for file in new_train_files]
        xml_paths = [os.path.join(get_images_from_folder_full, file) for file in new_train_files]

        count = 0
        for image_path, xml_path in zip(image_paths, xml_paths):
            original_image = Image.open(image_path).convert('RGB')
            original_targets = get_xml_data(xml_path)
            image_id = os.path.splitext(os.path.basename(image_path))[0]

            if count % 50 == 0:
                print(f"Processing training image {count}: {image_id}")
            # Generate proposals and targets
            proposal_images, proposal_targets = generate_proposals_and_targets_for_training(
                original_image, original_targets, transform, image_id,
                IOU_UPPER_LIMIT, IOU_LOWER_LIMIT, METHOD, MAX_PROPOSALS,
                generate_target=True
            )
            # Perform class balancing
            proposal_images_balanced, proposal_targets_balanced = class_balance(
                proposal_images, proposal_targets, SEED, image_id
            )

            if proposal_images_balanced is not None:
                pickle_save(
                    proposal_images_balanced, proposal_targets_balanced,
                    save_images_in_folder_full, save_targets_in_folder_full,
                    index=image_id, split='train'
                )
            count += 1

        logger.success("Training proposals and targets created successfully")

    if VALIDATION_PROPOSALS and new_val_files:
        logger.working_on(f"Creating validation proposals with {len(new_val_files)} images")
        image_paths_val = [os.path.join(get_images_from_folder_full, file.replace('.xml', '.jpg')) for file in new_val_files]
        xml_paths_val = [os.path.join(get_images_from_folder_full, file) for file in new_val_files]

        count = 0
        for image_path, xml_path in zip(image_paths_val, xml_paths_val):
            original_image = Image.open(image_path).convert('RGB')
            original_targets = get_xml_data(xml_path)
            image_id = os.path.splitext(os.path.basename(image_path))[0]

            if count % 50 == 0:
                print(f"Processing validation image {count}: {image_id}")

            # Generate proposals
            proposals = generate_proposals_for_test_and_val(
                original_image, original_targets, transform, image_id,
                IOU_UPPER_LIMIT, IOU_LOWER_LIMIT, METHOD, MAX_PROPOSALS,
                generate_target=True, return_images=False
            )

            if proposals is not None:
                # Save the proposals and image_id
                pickle_save(
                    None, proposals,
                    None, save_targets_in_folder_full_val,
                    index=image_id, split='val'
                )

                ground_truth_path = os.path.join(save_targets_in_folder_full_val, f"{image_id}_gt.pkl")
                save_ground_truth(ground_truth_path, original_targets)

                # Save the ground truth (original_targets)
                ground_truth_path = os.path.join(save_targets_in_folder_full_val, f"{image_id}_gt.pkl")


            count += 1

        logger.success("Validation proposals and ground truth saved successfully")
    if TEST_PROPOSALS:
        logger.working_on(f"Creating test proposals with {len(test_files)} images")
        image_paths_test = [os.path.join(get_images_from_folder_full, file.replace('.xml', '.jpg')) for file in test_files]
        xml_paths_test = [os.path.join(get_images_from_folder_full, file) for file in test_files]

        count = 0
        for image_path, xml_path in zip(image_paths_test, xml_paths_test):
            original_image = Image.open(image_path).convert('RGB')
            original_targets = get_xml_data(xml_path)
            image_id = os.path.splitext(os.path.basename(image_path))[0]

            if count % 50 == 0:
                print(f"Processing test image {count}: {image_id}")

            # Generate proposals
            proposals = generate_proposals_for_test_and_val(
                original_image, original_targets, transform, image_id,
                IOU_UPPER_LIMIT, IOU_LOWER_LIMIT, METHOD, MAX_PROPOSALS,
                generate_target=False, return_images=False
            )

            if proposals is not None:
                # Save the proposals and image_id
                pickle_save(
                    None, proposals,
                    None, save_targets_in_folder_full_test,
                    index=image_id, split='test'
                )

                # Save the ground truth (original_targets)
                ground_truth_path = os.path.join(save_targets_in_folder_full_test, f"{image_id}_gt.pkl")
                save_ground_truth(ground_truth_path, original_targets)

            count += 1

        logger.success("Test proposals and ground truth saved successfully")
