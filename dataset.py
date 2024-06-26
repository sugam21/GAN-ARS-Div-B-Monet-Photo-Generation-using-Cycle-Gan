from PIL import Image
import os
from torch.utils.data import Dataset
import numpy as np


class PhotoMonetDataset(Dataset):
    def __init__(
        self, root_dir_photo: str, root_dir_monet: str, transform=None
    ) -> None:
        self.root_dir_monet = root_dir_monet
        self.root_dir_photo = root_dir_photo
        self.transform = transform
        self.monet_images = os.listdir(self.root_dir_monet)
        self.photos = os.listdir(self.root_dir_photo)
        # since we need to have __len__ method and we might not have same length of images in both the dataset so we take the maximum of both
        self.length_dataset = max(len(self.monet_images), len(self.photos))

        self.monet_dataset_length = len(self.monet_images)
        self.photo_dataset_length = len(self.photos)

    def __len__(self):
        return self.length_dataset

    def __getitem__(self, idx):
        # it might happen that the index goes beyond and we get out of index error so to solve this we do modular division
        single_monet_image = self.monet_images[idx % self.monet_dataset_length]
        single_photo = self.photos[idx % self.photo_dataset_length]

        # the above is name of image only
        monet_complete_path = os.path.join(self.root_dir_monet, single_monet_image)
        photo_complete_path = os.path.join(self.root_dir_photo, single_photo)

        monet_image_arr = np.array(Image.open(monet_complete_path).convert("RGB"))
        photo_image_arr = np.array(Image.open(photo_complete_path).convert("RGB"))
        if self.transform:
            augmentations = self.transform(
                image=monet_image_arr, image0=photo_image_arr
            )
            photo_image_arr = augmentations["image0"]
            monet_image_arr = augmentations["image"]

        return photo_image_arr, monet_image_arr
