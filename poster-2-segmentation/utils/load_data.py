import glob
import torch
import os
import random

from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
import numpy as np

class PH2Dataset(Dataset):
    def __init__(self, split='train', transform=None, crop = False, data_path='/dtu/datasets1/02516/PH2_Dataset_images'):
        self.transform = transform
        self.image_paths = []
        self.data_path = data_path
        self.mask_paths = []
        self.crop = crop

        # Collect all IMDxxx directories
        sample_dirs = sorted(os.listdir(self.data_path))
        num_samples = len(sample_dirs)

        random.seed(42)  # For reproducibility
        random.shuffle(sample_dirs)

        train_split = int(0.7 * num_samples)
        val_split = int(0.85 * num_samples)

        if split == 'train':
            selected_dirs = sample_dirs[:train_split]
        elif split == 'val':
            selected_dirs = sample_dirs[train_split:val_split]
        elif split == 'test':
            selected_dirs = sample_dirs[val_split:]
        else:
            raise ValueError("split parameter should be 'train', 'val', or 'test'")

        for sample_dir in selected_dirs:
            sample_path = os.path.join(self.data_path, sample_dir)
            image_path = os.path.join(sample_path, f"{sample_dir}_Dermoscopic_Image", f"{sample_dir}.bmp")
            mask_path = os.path.join(sample_path, f"{sample_dir}_lesion", f"{sample_dir}_lesion.bmp")

            if os.path.exists(image_path) and os.path.exists(mask_path):
                self.image_paths.append(image_path)
                self.mask_paths.append(mask_path)
            else:
                print(f"No image or mask for {sample_dir}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')
        mask = Image.open(self.mask_paths[idx]).convert('L')  # Convert mask to grayscale

        if self.transform:
            if self.crop:
                image, mask = self.transform(image, mask)
            else:
                image = self.transform(image)

                transform_mask = transforms.Compose([
                    transforms.ToTensor(),
                ])

                mask = transform_mask(mask)

        # Binarize the mask (convert to 0s and 1s)
        mask = (mask > 0).float()  

        return image, mask

class DRIVEDataset(Dataset):
    def __init__(self, split='train', transform=None, crop = False, data_path='/dtu/datasets1/02516/DRIVE'):
        self.transform = transform
        self.data_path = data_path
        self.crop = crop

        data_type = 'training'
        images_dir = os.path.join(self.data_path, data_type, 'images')
        masks_dir = os.path.join(self.data_path, data_type, '1st_manual')

        image_paths = sorted(glob.glob(os.path.join(images_dir, '*_training.tif')))
        mask_paths = sorted(glob.glob(os.path.join(masks_dir, '*_manual1.gif')))

        assert len(image_paths) == len(mask_paths), "Mismatch between images and masks"

        combined = list(zip(image_paths, mask_paths))
        random.seed(42)
        random.shuffle(combined)
        image_paths[:], mask_paths[:] = zip(*combined)

        num_samples = len(image_paths)

        train_split = int(0.7 * num_samples)
        val_split = int(0.85 * num_samples)

        if split == 'train':
            self.image_paths = image_paths[:train_split]
            self.mask_paths = mask_paths[:train_split]
        elif split == 'val':
            self.image_paths = image_paths[train_split:val_split]
            self.mask_paths = mask_paths[train_split:val_split]
        elif split == 'test':
            self.image_paths = image_paths[val_split:]
            self.mask_paths = mask_paths[val_split:]
        else:
            raise ValueError("split parameter should be 'train', 'val', or 'test'")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')
        mask = Image.open(self.mask_paths[idx]).convert('L')  # Convert mask to grayscale

        if self.transform:
            if self.crop:
                image, mask = self.transform(image, mask)
            else:
                image = self.transform(image)

                transform_mask = transforms.Compose([
                    transforms.ToTensor(),
                ])

                mask = transform_mask(mask)

        # Binarize the mask (convert to 0s and 1s)
        mask = (mask > 0).float()  

        return image, mask

def load_data(data_name, split='train', transform=None, crop = False, data_path='/dtu/datasets1/02516', num_clicks=50, radius=20, seed=42, return_ground_truth=False, sampling='random'):
    if data_name.lower() == 'ph2':
        dataset = PH2Dataset(split=split, transform=transform, crop = crop, data_path=os.path.join(data_path, 'PH2_Dataset_images'))
    elif data_name.lower() == 'drive':
        dataset = DRIVEDataset(split=split, transform=transform, crop = crop, data_path=os.path.join(data_path, 'DRIVE'))
    elif data_name.lower() == 'ph2_weak_supervision':
        dataset = PH2DatasetWeakSupervision(split=split, transform=transform, crop = crop, data_path=os.path.join(data_path, 'PH2_Dataset_images'), num_clicks=num_clicks, radius=radius, seed=seed, return_ground_truth=False, sampling=sampling)
    else:
        raise ValueError(f"Dataset {data_name} not recognized.")
    return dataset

class PH2DatasetWeakSupervision(Dataset):
    def __init__(self, split='train', transform=None, crop=False, data_path='/dtu/datasets1/02516/PH2_Dataset_images', num_clicks=50, radius=10, seed=42, return_ground_truth=False, sampling='random'):
        self.transform = transform
        self.image_paths = []
        self.data_path = data_path
        self.mask_paths = []
        self.crop = crop
        self.num_clicks = num_clicks
        self.radius = radius
        self.seed = seed
        self.return_ground_truth = return_ground_truth
        self.sampling = sampling  # Store sampling method


        # Collect sample directories and perform train-val-test split
        sample_dirs = sorted(os.listdir(self.data_path))
        num_samples = len(sample_dirs)

        random.seed(seed)  # For reproducibility
        random.shuffle(sample_dirs)

        train_split = int(0.7 * num_samples)
        val_split = int(0.85 * num_samples)

        if split == 'train':
            selected_dirs = sample_dirs[:train_split]
        elif split == 'val':
            selected_dirs = sample_dirs[train_split:val_split]
        elif split == 'test':
            selected_dirs = sample_dirs[val_split:]
        else:
            raise ValueError("split parameter should be 'train', 'val', or 'test'")

        for sample_dir in selected_dirs:
            sample_path = os.path.join(self.data_path, sample_dir)
            image_path = os.path.join(sample_path, f"{sample_dir}_Dermoscopic_Image", f"{sample_dir}.bmp")
            mask_path = os.path.join(sample_path, f"{sample_dir}_lesion", f"{sample_dir}_lesion.bmp")

            if os.path.exists(image_path) and os.path.exists(mask_path):
                self.image_paths.append(image_path)
                self.mask_paths.append(mask_path)
            else:
                print(f"No image or mask for {sample_dir}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load the image and mask
        image = Image.open(self.image_paths[idx]).convert('RGB')
        mask = Image.open(self.mask_paths[idx]).convert('L')  # Convert mask to grayscale
        mask_np = (np.array(mask) > 0).astype(np.uint8)

        # generate the weak supervision mask based on the original mask
        if self.sampling == 'grid':
            new_mask_np = grid_sampling(mask_np, num_clicks_per_side=self.num_clicks, radius=self.radius)
        elif self.sampling == 'stratified':
            new_mask_np = stratified_sampling(mask_np, num_clicks_per_side=self.num_clicks, radius=self.radius)
        elif self.sampling == 'random':
            new_mask_np = add_points_randomMads(mask_np, num_clicks_per_side=self.num_clicks, radius=self.radius)
        else:
            raise ValueError(f"Sampling method '{self.sampling}' is not recognized.")

        #print(f'Unique values in new_mask_np: {np.unique(new_mask_np[~np.isnan(new_mask_np)])}')
        

        if self.transform:
            if self.crop:
                # Transform both image and new_mask together
                new_mask_pil = Image.fromarray(new_mask_np.astype(np.uint8))
                image, new_mask_pil = self.transform(image, new_mask_pil)
                new_mask_np = np.array(new_mask_pil, dtype=np.float32)
            else:
                image = self.transform(image)
                # Apply necessary transforms to new_mask_np if needed

        # Replace placeholder values (2) with NaN
        new_mask_np[new_mask_np == 2] = np.nan

        # Convert new_mask to tensor and add channel dimension
        new_mask = torch.from_numpy(new_mask_np)
        if new_mask.dim() == 2:
            new_mask = new_mask.unsqueeze(0)

        if self.return_ground_truth:
            # Load and process the full ground truth mask
            ground_truth_mask = torch.from_numpy(mask_np.astype(np.float32))
            if ground_truth_mask.dim() == 2:
                ground_truth_mask = ground_truth_mask.unsqueeze(0)
                # print the ground truth 
                #print(f'Unique values in ground_truth_mask: {np.unique(ground_truth_mask)}')
                print(ground_truth_mask)
            return image, new_mask, ground_truth_mask
        else:
            return image, new_mask

def load_data(data_name, split='train', transform=None, crop = False, data_path='/dtu/datasets1/02516', num_clicks=50, radius=20, seed=42, return_ground_truth=False, sampling='random'):
    if data_name.lower() == 'ph2':
        dataset = PH2Dataset(split=split, transform=transform, crop = crop, data_path=os.path.join(data_path, 'PH2_Dataset_images'))
    elif data_name.lower() == 'drive':
        dataset = DRIVEDataset(split=split, transform=transform, crop = crop, data_path=os.path.join(data_path, 'DRIVE'))
    elif data_name.lower() == 'ph2_weak_supervision':
        dataset = PH2DatasetWeakSupervision(split=split, transform=transform, crop = crop, data_path=os.path.join(data_path, 'PH2_Dataset_images'), num_clicks=num_clicks, radius=radius, seed=seed, return_ground_truth=False, sampling=sampling)
    else:
        raise ValueError(f"Dataset {data_name} not recognized.")
    return dataset

class PH2DatasetWeakSupervision(Dataset):
    def __init__(self, split='train', transform=None, crop=False, data_path='/dtu/datasets1/02516/PH2_Dataset_images', num_clicks=50, radius=10, seed=42, return_ground_truth=False, sampling='random'):
        self.transform = transform
        self.image_paths = []
        self.data_path = data_path
        self.mask_paths = []
        self.crop = crop
        self.num_clicks = num_clicks
        self.radius = radius
        self.seed = seed
        self.return_ground_truth = return_ground_truth
        self.sampling = sampling  # Store sampling method


        # Collect sample directories and perform train-val-test split
        sample_dirs = sorted(os.listdir(self.data_path))
        num_samples = len(sample_dirs)

        random.seed(seed)  # For reproducibility
        random.shuffle(sample_dirs)

        train_split = int(0.7 * num_samples)
        val_split = int(0.85 * num_samples)

        if split == 'train':
            selected_dirs = sample_dirs[:train_split]
        elif split == 'val':
            selected_dirs = sample_dirs[train_split:val_split]
        elif split == 'test':
            selected_dirs = sample_dirs[val_split:]
        else:
            raise ValueError("split parameter should be 'train', 'val', or 'test'")

        for sample_dir in selected_dirs:
            sample_path = os.path.join(self.data_path, sample_dir)
            image_path = os.path.join(sample_path, f"{sample_dir}_Dermoscopic_Image", f"{sample_dir}.bmp")
            mask_path = os.path.join(sample_path, f"{sample_dir}_lesion", f"{sample_dir}_lesion.bmp")

            if os.path.exists(image_path) and os.path.exists(mask_path):
                self.image_paths.append(image_path)
                self.mask_paths.append(mask_path)
            else:
                print(f"No image or mask for {sample_dir}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load the image and mask
        image = Image.open(self.image_paths[idx]).convert('RGB')
        mask = Image.open(self.mask_paths[idx]).convert('L')  # Convert mask to grayscale
        mask_np = (np.array(mask) > 0).astype(np.uint8)

        # generate the weak supervision mask based on the original mask
        if self.sampling == 'grid':
            new_mask_np = grid_sampling(mask_np, num_clicks_per_side=self.num_clicks, radius=self.radius)
        elif self.sampling == 'stratified':
            new_mask_np = stratified_sampling(mask_np, num_clicks_per_side=self.num_clicks, radius=self.radius)
        elif self.sampling == 'random':
            new_mask_np = add_points_randomMads(mask_np, num_clicks_per_side=self.num_clicks, radius=self.radius)
        else:
            raise ValueError(f"Sampling method '{self.sampling}' is not recognized.")

        #print(f'Unique values in new_mask_np: {np.unique(new_mask_np[~np.isnan(new_mask_np)])}')
        

        if self.transform:
            if self.crop:
                # Transform both image and new_mask together
                new_mask_pil = Image.fromarray(new_mask_np.astype(np.uint8))
                image, new_mask_pil = self.transform(image, new_mask_pil)
                new_mask_np = np.array(new_mask_pil, dtype=np.float32)
            else:
                image = self.transform(image)
                # Apply necessary transforms to new_mask_np if needed

        # Replace placeholder values (2) with NaN
        new_mask_np[new_mask_np == 2] = np.nan

        # Convert new_mask to tensor and add channel dimension
        new_mask = torch.from_numpy(new_mask_np)
        if new_mask.dim() == 2:
            new_mask = new_mask.unsqueeze(0)

        if self.return_ground_truth:
            # Load and process the full ground truth mask
            ground_truth_mask = torch.from_numpy(mask_np.astype(np.float32))
            if ground_truth_mask.dim() == 2:
                ground_truth_mask = ground_truth_mask.unsqueeze(0)
                # print the ground truth 
                #print(f'Unique values in ground_truth_mask: {np.unique(ground_truth_mask)}')
                print(ground_truth_mask)
            return image, new_mask, ground_truth_mask
        else:
            return image, new_mask
        

def add_points_randomMads(mask_array, num_clicks_per_side, radius):
    height, width = mask_array.shape
    new_mask = np.full((height, width), 2, dtype=np.uint8)  # 2 represents unknown
    np.random.seed(42)
    
    for _ in range(num_clicks_per_side):
        zero_indices = np.argwhere(mask_array == 0)
        one_indices = np.argwhere(mask_array == 1)
    
        if len(zero_indices) > 0:
            random_index = np.random.randint(len(zero_indices))
            selected_zero = zero_indices[random_index]
            new_mask = draw_circle(new_mask, selected_zero, radius, value=0)
    
        if len(one_indices) > 0:
            random_index = np.random.randint(len(one_indices))
            selected_one = one_indices[random_index]
            new_mask = draw_circle(new_mask, selected_one, radius, value=1)
    
    return new_mask  # [height, width]


def grid_sampling(mask_array, num_clicks_per_side, radius):
    """
    Adds points to the mask in a structured grid pattern.

    Parameters:
    - mask_array: 2D numpy array representing the original mask (values 0 and 1).
    - num_clicks_per_side: Number of grid divisions along one side (e.g., 10 means 10x10 grid).
    - radius: Radius of the circle to draw around each point.

    Returns:
    - new_mask: 2D numpy array with added annotations (0 for background, 1 for foreground, 2 for unknown).
    """
    height, width = mask_array.shape
    new_mask = np.full((height, width), 2, dtype=np.uint8)  # 2 represents unknown

    # Calculate the size of each grid cell
    cell_height = height // num_clicks_per_side
    cell_width = width // num_clicks_per_side

    for i in range(num_clicks_per_side):
        for j in range(num_clicks_per_side):
            # Calculate the center of the current cell
            center_y = i * cell_height + cell_height // 2
            center_x = j * cell_width + cell_width // 2

            # Ensure the center point is within the bounds of the mask
            if center_y < height and center_x < width:
                # Check the original mask value at this center point
                if mask_array[center_y, center_x] == 1:
                    new_mask = draw_circle(new_mask, (center_y, center_x), radius, value=1)
                else:
                    new_mask = draw_circle(new_mask, (center_y, center_x), radius, value=0)

    return new_mask

def stratified_sampling(mask_array, num_clicks_per_side, radius):
    """
    Adds points to the mask using stratified sampling.

    Parameters:
    - mask_array: 2D numpy array representing the original mask (values 0 and 1).
    - total_num_clicks: Total number of clicks to distribute between classes.
    - radius: Radius of the circle to draw around each point.

    Returns:
    - new_mask: 2D numpy array with added annotations (0 for background, 1 for foreground, 2 for unknown).
    """
    height, width = mask_array.shape
    new_mask = np.full((height, width), 2, dtype=np.uint8)  # 2 represents unknown
    np.random.seed(42)

    # Calculate the number of pixels in each class
    num_zero = np.sum(mask_array == 0)
    num_one = np.sum(mask_array == 1)
    total_pixels = num_zero + num_one

    # Calculate the number of clicks for each class proportionally
    num_clicks_zero = int((num_zero / total_pixels) * num_clicks_per_side)
    num_clicks_one = num_clicks_per_side - num_clicks_zero  # Remaining clicks

    # Get indices of each class
    zero_indices = np.argwhere(mask_array == 0)
    one_indices = np.argwhere(mask_array == 1)

    # Sample points for background
    if len(zero_indices) > 0 and num_clicks_zero > 0:
        sampled_zero_indices = zero_indices[np.random.choice(len(zero_indices), num_clicks_zero, replace=False)]
        for idx in sampled_zero_indices:
            new_mask = draw_circle(new_mask, idx, radius, value=0)

    # Sample points for foreground
    if len(one_indices) > 0 and num_clicks_one > 0:
        sampled_one_indices = one_indices[np.random.choice(len(one_indices), num_clicks_one, replace=False)]
        for idx in sampled_one_indices:
            new_mask = draw_circle(new_mask, idx, radius, value=1)

    return new_mask  # [height, width]

def draw_circle(array, center, radius, value):
    """
    Draws a filled circle on a numpy array.

    Parameters:
    - array: 2D numpy array.
    - center: Tuple of (y, x) coordinates for the center of the circle.
    - radius: Radius of the circle.
    - value: Value to assign inside the circle.

    Returns:
    - array: Modified array with the circle drawn.
    """
    height, width = array.shape
    y, x = center
    Y, X = np.ogrid[:height, :width]
    dist_from_center = (X - x)**2 + (Y - y)**2
    mask = dist_from_center <= radius**2
    array[mask] = value
    return array

def add_points_MADS(mask_array, num_clicks_per_side, radius):
    """

    """
    height, width = mask_array.shape
    new_mask = np.full((height, width), 2, dtype=np.uint8)  # 2 represents unknown

    # Calculate the size of each grid cell
    cell_height = height // num_clicks_per_side
    cell_width = width // num_clicks_per_side

    for i in range(num_clicks_per_side):
        for j in range(num_clicks_per_side):
            # Calculate the center of the current cell
            center_y = i * cell_height + cell_height // 2
            center_x = j * cell_width + cell_width // 2

            # Ensure the center point is within the bounds of the mask
            if center_y < height and center_x < width:
                # Check the original mask value at this center point
                if mask_array[center_y, center_x] == 1:
                    new_mask = draw_circle(new_mask, (center_y, center_x), radius, value=1)
                else:
                    new_mask = draw_circle(new_mask, (center_y, center_x), radius, value=0)

    return new_mask

if __name__ == "__main__":
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])

    # PH2 Dataset
    print("===== Loading PH2 data =====")
    ph2_train_dataset = load_data('ph2', split='train', transform=transform)
    ph2_val_dataset = load_data('ph2', split='val', transform=transform)
    ph2_test_dataset = load_data('ph2', split='test', transform=transform)

    # Data loaders
    ph2_train_loader = torch.utils.data.DataLoader(ph2_train_dataset, batch_size=16, shuffle=True)
    ph2_val_loader = torch.utils.data.DataLoader(ph2_val_dataset, batch_size=16, shuffle=False)
    ph2_test_loader = torch.utils.data.DataLoader(ph2_test_dataset, batch_size=16, shuffle=False)


    image, mask = ph2_train_dataset[1]
    print('Image shape:', image.shape)
    print('Mask shape:', mask.shape)

    print("===== Loading DRIVE data =====")
    # DRIVE Dataset
    drive_train_dataset = load_data('drive', split='train', transform=transform)
    drive_val_dataset = load_data('drive', split='val', transform=transform)
    drive_test_dataset = load_data('drive', split='test', transform=transform)

    # Data loaders
    drive_train_loader = torch.utils.data.DataLoader(drive_train_dataset, batch_size=4, shuffle=True)
    drive_val_loader = torch.utils.data.DataLoader(drive_val_dataset, batch_size=4, shuffle=False)
    drive_test_loader = torch.utils.data.DataLoader(drive_test_dataset, batch_size=4, shuffle=False)

    image, mask = drive_train_dataset[1]
    print('Image shape:', image.shape)
    print('Mask shape:', mask.shape)

    print("Data loaded")