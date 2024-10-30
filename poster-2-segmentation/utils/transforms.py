import torchvision.transforms.functional as TF
import torchvision.transforms as T
import torch
import numpy as np
from PIL import Image

class JointTransform:
    def __init__(self, crop_size=None, resize=None, mean = None, std = None):
        self.crop_size = crop_size
        self.resize = resize
        self.mean = mean
        self.std = std

        if mean is not None and std is not None:
            self.normalize = T.Normalize(mean=mean, std=std)
        else:
            self.normalize = None

    def __call__(self, image, mask):
        # Resize
        if self.resize is not None:
            image = TF.resize(image, self.resize)
            mask = TF.resize(mask, self.resize)

        # Random Crop
        if self.crop_size is not None:
            i, j, h, w = T.RandomCrop.get_params(image, output_size=self.crop_size)
            image = TF.crop(image, i, j, h, w)
            mask = TF.crop(mask, i, j, h, w)


        # Other transformations (if any) can be added here

        # Convert to Tensor
        image = TF.to_tensor(image)
        mask = TF.to_tensor(mask)
        # Normalize image
        if self.normalize is not None:
            image = self.normalize(image)

        return image, mask

class JointTransform_weak:
    def __init__(self, crop_size=None, resize=None, mean = None, std=None):
        self.crop_size = crop_size
        self.resize = resize
        self.mean = mean
        self.std = std

        if mean is not None and std is not None:
            self.normalize = T.Normalize(mean=mean, std=std)
        else:
            self.normalize = None

    def __call__(self, image, mask):
        # Random crop
        if self.crop_size is not None:
            i, j, h, w = T.RandomCrop.get_params(image, output_size=self.crop_size)
            image = TF.crop(image, i, j, h, w)
            mask = TF.crop(mask, i, j, h, w)

        # Resize
        if self.resize is not None:
            image = TF.resize(image, self.resize)
            mask = TF.resize(mask, self.resize, interpolation=Image.NEAREST)

        # Convert image to tensor

        image = TF.to_tensor(image)

        # Convert mask to tensor
        mask = torch.from_numpy(np.array(mask, dtype=np.float32))

        # Ensure mask has shape [1, H, W]
        if mask.dim() == 2:
            mask = mask.unsqueeze(0)

        # Normalize image
        if self.normalize is not None:
            image = self.normalize(image)

        return image, mask