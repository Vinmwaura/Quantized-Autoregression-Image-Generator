import os
import json
import random
import pathlib
import argparse

import numpy as np
from tinydb import TinyDB

import torch

from models.FC_Encoder import FC_Encoder
from dataset_loader.image_dataset import ImageDataset

# Utility functions.
from utils.model_utils import load_model

# Save encodings from Encoder into memory for easier retrieval and computation.
def save_feature_maps(
        model,
        dataloader,
        out_dir,
        device="cpu",
        num_files_folder=1_000):
    file_index = 0
    folder_name = 0

    print("#" * 100)
    print("Saving Feature Maps to disk...")

    all_data = []
    for index, (image, image_paths) in enumerate(dataloader):
        image = image.to(device)  # (N, C, H, W)
        with torch.no_grad():
            latent_maps = model(image)  # (N, C_map, H_map, W_map)
        latent_maps_cpu = latent_maps.cpu()

        for feature_map, image_path in zip(latent_maps_cpu, image_paths):
            if file_index % num_files_folder == 0 and file_index > 0:
                folder_name = folder_name + 1

            curr_folder = os.path.join(out_dir, str(folder_name))
            os.makedirs(curr_folder, exist_ok=True)

            feature_map_path = os.path.join(
                curr_folder,
                f"{file_index}")

            feature_map_numpy = feature_map.numpy()

            with open(feature_map_path, 'wb') as f:
                np.save(
                    f,
                    feature_map_numpy,
                    allow_pickle=False,
                    fix_imports=False)

            file_index = file_index + 1

            all_data.append({
                "fmap_path": feature_map_path,
                "image_path": image_path
            })

        print(f"{(index + 1):,} / {len(dataloader):,}")
    print("Finished saving feature maps.")

    all_db_filename = os.path.join(
        out_dir,
        "all_dataset.json")
    db = TinyDB(all_db_filename)
    db.insert_multiple(all_data)
    print("Finished saving json file.")
    print("#" * 100)

def main():
    parser = argparse.ArgumentParser(
        description="Generate Feature Maps Dataset.")

    parser.add_argument(
        "--device",
        help="Which hardware device will model run on (default='cpu')?",
        choices=['cpu', 'cuda'],
        type=str,
        default="cpu")
    parser.add_argument(
        "--batch-size",
        help="Batch size for dataset.",
        type=int,
        default=8)
    parser.add_argument(
        "--num-files-folder",
        help="Number of files per folder.",
        type=int,
        default=1_000)
    parser.add_argument(
        "--dataset-path",
        help="File path to image dataset json file.",
        required=True,
        type=pathlib.Path)
    parser.add_argument(
        "--model-path",
        help="File path to saved Encoder model checkpoint",
        required=True,
        type=pathlib.Path)
    parser.add_argument(
        "--out-dir",
        help="File path to output directory",
        required=True,
        type=pathlib.Path)

    args = vars(parser.parse_args())

    device = args["device"]  # Device to run model on.
    model_path = args["model_path"]  # File path to model.
    dataset_path = args["dataset_path"]  # File path to dataset.
    batch_size = args["batch_size"]  # Batch size.
    num_files_folder = args["num_files_folder"]  # Max number of files in each folder.
    out_dir = args["out_dir"]  # Output directory.
    try:
        os.makedirs(out_dir, exist_ok=True)
    except Exception as e:
        raise e

    encoder_status, encoder_dict = load_model(model_path)
    if not encoder_status:
        raise Exception("An error occured while loading Encoder model checkpoint!")

    # Autoencoder Params.
    image_channel = encoder_dict["image_channel"]
    min_channel = encoder_dict["min_channel"]
    max_channel = encoder_dict["max_channel"]
    num_layers = encoder_dict["num_layers"]
    latent_channel = encoder_dict["latent_channel"]
    hidden_activation_type = encoder_dict["hidden_activation_type"]
    use_final_activation = encoder_dict["use_final_dec_activation"]
    encoder_activation_type = encoder_dict["encoder_activation_type"]

    # Encoder model
    encoder = FC_Encoder(
        num_layers=num_layers,
        image_channel=image_channel,
        min_channel=min_channel,
        max_channel=max_channel,
        latent_channel=latent_channel,
        hidden_activation_type=hidden_activation_type,
        use_final_activation=use_final_activation,
        final_activation_type=encoder_activation_type)

    encoder.custom_load_state_dict(encoder_dict["model"])

    encoder = encoder.to(device)

    # Load all Image Dataset.
    img_dataset = ImageDataset(
        dataset_path=dataset_path,
        return_filepaths=True)
    img_dataloader = torch.utils.data.DataLoader(
        img_dataset,
        batch_size=batch_size,
        num_workers=4,
        shuffle=True)

    # Save feature maps to storage for further computations.
    save_feature_maps(
        model=encoder,
        dataloader=img_dataloader,
        out_dir=out_dir,
        device=device,
        num_files_folder=num_files_folder)

if __name__ == "__main__":
    main()
