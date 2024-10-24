import glob
import torch
import os
import random

from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms

class PH2Dataset(Dataset):
    def __init__(self, split='train', transform=None, data_path='/dtu/datasets1/02516/PH2_Dataset_images'):
        self.transform = transform
        self.image_paths = []
        self.data_path = data_path
        self.mask_paths = []

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
            image, mask = self.transform(image, mask)

        return image, mask

class DRIVEDataset(Dataset):
    def __init__(self, split='train', transform=None, data_path='/dtu/datasets1/02516/DRIVE'):
        self.transform = transform
        self.data_path = data_path

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
        mask = Image.open(self.mask_paths[idx]).convert('L')

        if self.transform:
            image, mask = self.transform(image, mask)

        return image, mask


def load_data(data_name, split='train', transform=None, data_path='/dtu/datasets1/02516'):
    if data_name.lower() == 'ph2':
        dataset = PH2Dataset(split=split, transform=transform, data_path=os.path.join(data_path, 'PH2_Dataset_images'))
    elif data_name.lower() == 'drive':
        dataset = DRIVEDataset(split=split, transform=transform, data_path=os.path.join(data_path, 'DRIVE'))
    else:
        raise ValueError(f"Dataset {data_name} not recognized.")
    return dataset


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