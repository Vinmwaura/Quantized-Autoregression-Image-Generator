import cv2
import numpy as np

from tinydb import TinyDB

import torch
from torch.utils.data import Dataset

"""
Loads Feature Maps Dataset.
"""
class FeatureMapDataset(Dataset):
    def __init__(
            self,
            dataset_path,
            load_image=False,
            return_filepaths=False):
        self.load_image = load_image
        self.return_filepaths = return_filepaths

        # Load Tinydb dataset.
        self.db = TinyDB(dataset_path)
        self.data_list = self.db.all()

        # Check if data is not empty.
        if len(self.data_list) == 0:
            raise Exception("No data found.")

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        data_dict = self.data_list[index]

        fmap_path = data_dict["fmap_path"]

        # Load map from storage.
        with open(fmap_path, "rb") as f:
            fmap = np.load(f)

        # Convert image as numpy to Tensor.
        fmap_tensor = torch.from_numpy(fmap).float()
    
        if self.load_image:
            image_path = data_dict["image_path"]

            # Load images using opencv2.
            image = cv2.imread(image_path)  # (H, W, C)

            # Scale images to be between 1 and -1.
            image = (image.astype(float) - 127.5) / 127.5

            # Convert image as numpy to Tensor.
            image_tensor = torch.from_numpy(image).float()

            if self.return_filepaths:
                return fmap_tensor, fmap_path, image_tensor, image_path

            return fmap_tensor, image_tensor

        if self.return_filepaths:
            return fmap_tensor, fmap_path

        return fmap_tensor
