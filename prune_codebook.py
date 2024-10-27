import os
import json
import pathlib
import logging
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.Codebook import Codebook

from dataset_loader.feature_map_dataset import FeatureMapDataset

# Utility functions.
from utils.image_utils import save_images
from utils.model_utils import (
    save_model,
    load_model)

def main():
    project_name = "Prune Codebook"

    parser = argparse.ArgumentParser(
        description=f"Train {project_name}.")

    parser.add_argument(
        "--device",
        help="Which hardware device will model run on.",
        choices=['cpu', 'cuda'],
        type=str,
        default="cpu")
    parser.add_argument(
        "--dataset-path",
        help="File path to image dataset json file.",
        required=True,
        type=pathlib.Path)
    parser.add_argument(
        "--codebook-path",
        help="File path to saved codebook.",
        required=True,
        type=pathlib.Path)
    parser.add_argument(
        "--batch-size",
        help="Batch size for dataset.",
        type=int,
        default=8)
    parser.add_argument(
        "--prune-threshold",
        help="Threshold value below which to prune embeddings.",
        type=int,
        default=10)
    parser.add_argument(
        "--out-dir",
        help="File path to output directory.",
        required=True,
        type=pathlib.Path)

    args = vars(parser.parse_args())

    device = args["device"]  # Device to run model on.
    codebook_path = args["codebook_path"]  # File path to Autoencoder model.
    dataset_path = args["dataset_path"]  # File path to dataset.
    batch_size = args["batch_size"]  # Batch size.
    prune_threshold = args["prune_threshold"]
    out_dir = args["out_dir"]  # Output Directory.
    try:
        os.makedirs(out_dir, exist_ok=True)
    except Exception as e:
        raise e

    log_path = os.path.join(out_dir, f"{project_name}.log")
    logging.basicConfig(
        # filename=log_path,
        format="%(asctime)s %(message)s",
        encoding='utf-8',
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler()
        ],
        level=logging.DEBUG)

    codebook_status, codebook_dict = load_model(codebook_path)
    if not codebook_status:
        raise Exception("An error occured while loading codebook checkpoint!")

    codebook_patch_dim = codebook_dict["patch_dim"]
    codebook_image_dim = codebook_dict["image_dim"]
    codebook_image_channel = codebook_dict["image_C"]
    codebook_num_embeddings = codebook_dict["num_embeddings"]
    codebook_neighbourhood_range = codebook_dict["neighbourhood_range"]
    
    logging.info(f"{project_name}")
    logging.info(f"Output Dir: {out_dir}")
    logging.info("#" * 100)
    logging.info(f"Codebook Parameters.")
    logging.info(f"Image dim: {codebook_image_dim}")
    logging.info(f"Image channel: {codebook_image_channel:,}")
    logging.info(f"Patch size: {codebook_patch_dim}")
    logging.info(f"Num Embeddings: {codebook_num_embeddings:,}")
    logging.info(f"Neighbourhood range: {codebook_neighbourhood_range:,}")
    logging.info("#" * 100)

    # Codebook.
    codebook = Codebook(
        patch_dim=codebook_patch_dim,
        image_dim=codebook_image_dim,
        image_channel=codebook_image_channel,
        num_embeddings=codebook_num_embeddings,
        init_neighbour_range=codebook_neighbourhood_range)

    codebook.custom_load_state_dict(codebook_dict["checkpoint"])

    codebook = codebook.to(device)

    global_steps = codebook_dict["global_steps"]

    # Image Dataset.
    feature_map_dataset = FeatureMapDataset(
        dataset_path=dataset_path,
        load_image=False,
        return_filepaths=False)
    feature_map_dataloader = torch.utils.data.DataLoader(
        feature_map_dataset,
        batch_size=batch_size,
        num_workers=4,
        shuffle=True)

    total_embedding_indices = {}
    for i in range(codebook_num_embeddings):
        total_embedding_indices[i] = 0

    for index, feature_map in enumerate(feature_map_dataloader):
        feature_map = feature_map.to(device)

        codebook.eval()

        bmu_indices = codebook.get_patches_bmu(feature_map)  # (N*Seq)
        bmu_indices_list = bmu_indices.tolist()

        for bmu_index in bmu_indices_list:
            total_embedding_indices[bmu_index] += 1

    good_embeddings = []
    for i, count in total_embedding_indices.items():
        print(f"{i}: {count:,}")
        if count >= prune_threshold:
            good_embeddings.append(i)
    logging.info(f"Saved embeddings: {len(good_embeddings)}")

    # New Codebook.
    new_codebook_num_embeddings = len(good_embeddings)
    new_codebook = Codebook(
        patch_dim=codebook_patch_dim,
        image_dim=codebook_image_dim,
        image_channel=codebook_image_channel,
        num_embeddings=new_codebook_num_embeddings,
        init_neighbour_range=codebook_neighbourhood_range)

    # Copy the weights of the embeddings that are kept
    with torch.no_grad():
        new_codebook.codebook.weight.copy_(codebook.codebook.weight[good_embeddings])

    # Save Codebook.
    codebook_dict = {
        "patch_dim": codebook_patch_dim,
        "image_dim": codebook_image_dim,
        "image_C": codebook_image_channel,
        "num_embeddings": new_codebook_num_embeddings,
        "neighbourhood_range": codebook_neighbourhood_range,
        "global_steps": global_steps,
        "checkpoint": new_codebook.state_dict()}

    save_status = save_model(
        model_dict=codebook_dict,
        dest_path=out_dir,
        file_name=f"pruned_codebook.pt",
        logging=logging.info)
    if save_status is True:
        logging.info("Successfully saved codebook.")
    else:
        logging.info("Error occured saving codebook.")

if __name__ == "__main__":
    main()
