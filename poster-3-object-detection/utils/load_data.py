
import os
import glob
import torch

from PIL import Image
from torch.utils.data import Dataset



class Potholes(Dataset):
    def __init__(self, transform=None, folder_path='Potholes/annotated-images'):
        self.folder_path = folder_path
        self.transform = transform

        #Return a list of paths
        self.image_paths = glob.glob(os.path.join(folder_path, "img-*.jpg"))
        self.xml_paths = glob.glob(os.path.join(folder_path, "img-*.xml"))
        assert len(self.image_paths) == len(self.xml_paths), 'Number of images and xml files does not match'



        for image_path, xml_path in zip(self.image_paths, self.xml_paths):
            None


    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image


def main():

    c = Potholes(folder_path='Potholes/annotated-images')
    print(c.__getitem__(1))
    





if __name__ == "__main__":
    main()