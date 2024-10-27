import os

import torch
import torchvision

# Save tensor images as a grid.
def save_images(
        images,
        file_name,
        dest_path,
        nrow=5,
        logging=print):
    try:
        # Convert from BGR to RGB,
        permute_dim = [2, 1, 0]
        images = images[:, permute_dim]

        # Save images in a grid.
        grid_img = torchvision.utils.make_grid(
            images,
            nrow=nrow,
            normalize=True,
            value_range=(-1, 1))

        dir_path = os.path.join(
            dest_path,
            "images")

        # Create folder if doesn't exist.
        os.makedirs(dir_path, exist_ok=True)

        path = os.path.join(
            dir_path,
            str(file_name) + ".jpg")
        torchvision.utils.save_image(
            grid_img,
            path)
        logging(f"Saving image: {path}")
        success = True
    except Exception as e:
        logging(f"An error occured while saving image: {e}")
        success = False
    finally:
        return success
