import os
import json
import pathlib
import logging
import argparse

import torch
import torch.nn.functional as F

from models.Codebook import Codebook
from models.FC_Decoder import FC_Decoder

from dataset_loader.feature_map_dataset import FeatureMapDataset

# Utility functions.
from utils.image_utils import save_images
from utils.model_utils import (
    save_model,
    load_model)


def main():
    project_name = "Codebook"

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
        "--decoder-path",
        help="File path to pre-trained decoder model.",
        required=True,
        type=pathlib.Path)
    parser.add_argument(
        "--codebook-path",
        help="File path to saved codebook.",
        required=False,
        type=pathlib.Path)
    parser.add_argument(
        "--batch-size",
        help="Batch size for dataset.",
        type=int,
        default=8)
    parser.add_argument(
        "--checkpoint-step",
        help="Steps at which checkpoint takes place.",
        type=int,
        default=1_000)
    parser.add_argument(
        "--lr-step",
        help="Steps before halving learning rate.",
        type=int,
        default=100_000)
    parser.add_argument(
        "--max-epoch",
        help="Maximum epoch for training model.",
        type=int,
        default=1_000)
    parser.add_argument(
        "-c",
        "--config-path",
        help="File path to load json config file.",
        required=True,
        type=pathlib.Path)
    parser.add_argument(
        "--out-dir",
        help="File path to output directory.",
        required=True,
        type=pathlib.Path)

    args = vars(parser.parse_args())

    # Load and Parse config JSON.
    config_json = args["config_path"]
    with open(config_json, 'r') as json_file:
        json_data = json_file.read()
    config_dict = json.loads(json_data)

    device = args["device"]  # Device to run model on.
    codebook_path = args["codebook_path"]  # File path to Autoencoder model.
    dataset_path = args["dataset_path"]  # File path to dataset.
    batch_size = args["batch_size"]  # Batch size.
    decoder_path = args["decoder_path"]
    model_checkpoint_step = args["checkpoint_step"]  # Number of steps to checkpoint.
    lr_update_step = args["lr_step"]  # Number of steps before halving learning rate.
    max_epoch = args["max_epoch"]  # Max epoch for training.
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

    # Training Params.
    curr_epoch = 0
    global_steps = 0

    # Load Pre-trained Decoder model.
    decoder_model_status, decoder_model_dict = load_model(decoder_path)
    if not decoder_model_status:
        raise Exception("An error occured while loading decoder model checkpoint!")

    dec_num_layers = decoder_model_dict["num_layers"]
    dec_image_channel = decoder_model_dict["image_channel"]
    dec_min_channel = decoder_model_dict["min_channel"]
    dec_max_channel = decoder_model_dict["max_channel"]
    dec_latent_channel = decoder_model_dict["latent_channel"]
    dec_hidden_activation_type = decoder_model_dict["hidden_activation_type"]
    dec_use_final_activation = decoder_model_dict["use_final_dec_activation"]
    dec_final_activation_type = decoder_model_dict["decoder_activation_type"]

    decoder_model = FC_Decoder(
        num_layers=dec_num_layers,
        image_channel=dec_image_channel,
        min_channel=dec_min_channel,
        max_channel=dec_max_channel,
        latent_channel=dec_latent_channel,
        hidden_activation_type=dec_hidden_activation_type,
        use_final_activation=dec_use_final_activation,
        final_activation_type=dec_final_activation_type).to(device)
    decoder_model.custom_load_state_dict(decoder_model_dict["model"])

    # Model Params.
    model_lr = config_dict["model_lr"]
    neighbourhood_step = config_dict["neighbourhood_step"]

    if codebook_path is not None:
        codebook_status, codebook_dict = load_model(codebook_path)
        if not codebook_status:
            raise Exception("An error occured while loading codebook checkpoint!")

        patch_dim = codebook_dict["patch_dim"]
        num_embeddings = codebook_dict["num_embeddings"]
        image_dim = codebook_dict["image_dim"]
        image_C = codebook_dict["image_C"]

        # Initialize Codebook.
        codebook = Codebook(
            patch_dim=patch_dim,
            image_dim=image_dim,
            image_channel=image_C,
            num_embeddings=num_embeddings,
            init_neighbour_range=codebook_dict["neighbourhood_range"])

        codebook.custom_load_state_dict(codebook_dict["checkpoint"])

        global_steps = codebook_dict["global_steps"]
    else:
        image_dim = (config_dict["image_H"], config_dict["image_W"])
        image_C = config_dict["image_C"]
        patch_dim = (config_dict["patch_H"], config_dict["patch_W"]) 
        num_embeddings = config_dict["num_embeddings"]

        # Initialize Codebook.
        codebook = Codebook(
            patch_dim=patch_dim,
            image_dim=image_dim,
            image_channel=image_C,
            num_embeddings=num_embeddings,
            init_neighbour_range=num_embeddings//2)

    codebook = codebook.to(device)
    codebook_optim = torch.optim.Adam(
        codebook.parameters(),
        lr=model_lr,
        betas=(0.5, 0.999))
    
    # Feature Map Dataset.
    feature_map_dataset = FeatureMapDataset(
        dataset_path=dataset_path,
        load_image=False,
        return_filepaths=False)
    feature_map_dataloader = torch.utils.data.DataLoader(
        feature_map_dataset,
        batch_size=batch_size,
        num_workers=4,
        shuffle=True)

    logging.info(f"{project_name}")
    logging.info(f"Output Dir: {out_dir}")
    logging.info("#" * 100)
    logging.info(f"Codebook Parameters.")
    logging.info(f"Image dim: {image_dim}")
    logging.info(f"Image channel: {image_C:,}")
    logging.info(f"Patch size: {codebook.patch_dim}")
    logging.info(f"Num Embeddings: {codebook.num_embeddings:,}")
    logging.info(f"Neighbourhood range: {codebook.neighbourhood_range:,}")
    logging.info("#" * 100)
    logging.info(f"Training Parameters.")
    logging.info(f"Max Epoch: {max_epoch:,}")
    logging.info(f"Batch Size: {batch_size:,}")
    logging.info(f"Model LR Update size: {lr_update_step:,}")
    logging.info(f"Model Checkpoint step: {model_checkpoint_step:,}")
    logging.info("#" * 100)

    for epoch in range(curr_epoch, max_epoch):
        iteration_count = 0
        total_recon_loss = 0

        for index, feature_map in enumerate(feature_map_dataloader):
            iteration_count = iteration_count + 1

            feature_map = feature_map.to(device)

            codebook.train()

            codebook_optim.zero_grad()

            feature_map_quant = codebook(
                feature_map,
                use_gaussian=True)

            recon_loss = F.mse_loss(
                feature_map_quant,
                feature_map)

            if torch.isnan(recon_loss):
                raise Exception("NaN encountered during training")

            recon_loss.backward()

            codebook_optim.step()

            total_recon_loss += recon_loss.item()

            # Half learning rate.
            if global_steps % lr_update_step == 0 and global_steps > 0:
                for codebook_params in codebook_optim.param_groups:
                    codebook_params['lr'] = codebook_params['lr'] * 0.5

            # Model checkpoints.
            if global_steps % model_checkpoint_step == 0:
                decoder_model.eval()

                with torch.no_grad():
                    test_dec_image = decoder_model(feature_map)
                    test_dec_quant = decoder_model(feature_map_quant)

                _ = save_images(
                    images=test_dec_image,
                    file_name=f"image_plot_{global_steps}",
                    dest_path=out_dir,
                    logging=logging.info)
                _ = save_images(
                    images=test_dec_quant,
                    file_name=f"quant_image_plot_{global_steps}",
                    dest_path=out_dir,
                    logging=logging.info)

                # Save Codebook.
                codebook_dict = {
                    "patch_dim": patch_dim,
                    "image_dim": image_dim,
                    "image_C": image_C,
                    "num_embeddings": codebook.num_embeddings,
                    "neighbourhood_range": codebook.neighbourhood_range,
                    "global_steps": global_steps,
                    "checkpoint": codebook.state_dict()}

                save_status = save_model(
                    model_dict=codebook_dict,
                    dest_path=out_dir,
                    file_name=f"codebook_{global_steps}.pt",
                    logging=logging.info)
                if save_status is True:
                    logging.info("Successfully saved codebook.")
                else:
                    logging.info("Error occured saving codebook.")

            temp_avg_recon = total_recon_loss / iteration_count
            message = "Cum. Steps: {:,} | Steps: {:,} / {:,} | L.R.: {:.8f} | Recon Loss: {:.5f} | Neighbourhood Range: {}".format(
                global_steps + 1,
                index + 1,
                len(feature_map_dataloader),
                codebook_optim.param_groups[0]['lr'],
                temp_avg_recon,
                codebook.neighbourhood_range)
            logging.info(message)

            global_steps += 1

            if global_steps % neighbourhood_step == 0:
                # Decrease Neighbourhood range after set step.
                codebook.decrease_neighbourhood(steps=1)

if __name__ == "__main__":
    main()
