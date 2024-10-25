import torchvision.transforms.functional as TF
import torchvision.transforms as T

import random

class JointTransform:
    def __init__(self, crop_size=None, resize=None):
        self.crop_size = crop_size
        self.resize = resize

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

        return image, mask