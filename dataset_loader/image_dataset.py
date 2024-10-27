import cv2
from tinydb import TinyDB

import torch
from torch.utils.data import Dataset


"""
Load Images and optionally filepaths.
"""
class ImageDataset(Dataset):
    def __init__(
            self,
            dataset_path,
            return_filepaths=False):
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

        image_fpath = data_dict["image_fpath"]

        # Load images using opencv2.
        image = cv2.imread(image_fpath)  # (H, W, C)

        # Scale images to be between 1 and -1.
        image = (image.astype(float) - 127.5) / 127.5

        # Convert image as numpy to Tensor.
        image_tensor = torch.from_numpy(image).float()

        # Permute image to be of format: [C,H,W]
        image_tensor = image_tensor.permute(2, 0, 1)

        if self.return_filepaths:
            return image_tensor, image_fpath

        return image_tensor
