import os
import json
import math
import random
import pathlib
import logging
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.Autoencoder import Autoencoder

from dataset_loader.image_dataset import ImageDataset

# Utility functions.
from utils.model_utils import (
    save_model,
    load_model)
from utils.image_utils import save_images

def main():
    project_name = "Autoencoder"

    parser = argparse.ArgumentParser(
        description=f"Train {project_name} models.")
    
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
        "--model-path",
        help="File path to saved model checkpoint.",
        default=None,
        required=False,
        type=pathlib.Path)
    parser.add_argument(
        "--load-optim",
        action='store_true',
        help="Load saved optim parameters with model.")
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
        default=50_000)
    parser.add_argument(
        "--max-epoch",
        help="Maximum epoch for training model.",
        type=int,
        default=1_000)
    parser.add_argument(
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
    model_path = args["model_path"]  # File path to model.
    load_optim = args["load_optim"]  # Load saved optim parameters.
    dataset_path = args["dataset_path"]  # File path to dataset.
    batch_size = args["batch_size"]  # Batch size.
    model_checkpoint_step = args["checkpoint_step"]  # Number of steps to checkpoint.
    lr_update_step = args["lr_step"]  # Number of steps before halving learning rate.
    max_epoch = args["max_epoch"]  # Max epoch for training.
    out_dir = args["out_dir"]  # Output Directory.
    try:
        os.makedirs(out_dir, exist_ok=True)
    except Exception as e:
        raise e

    # Training Params.
    curr_epoch = 0
    global_steps = 0

    # Model Params.
    model_lr = config_dict["model_lr"]
    num_layers = config_dict["num_layers"]
    image_channel = config_dict["image_channel"]
    min_channel = config_dict["min_channel"]
    max_channel = config_dict["max_channel"]
    latent_channel = config_dict["latent_channel"]
    hidden_activation_type = config_dict["hidden_activation_type"]
    use_final_enc_activation = config_dict["use_final_enc_activation"]
    encoder_activation_type = "silu" if not use_final_enc_activation else config_dict["encoder_activation_type"]
    use_final_dec_activation = config_dict["use_final_dec_activation"]
    decoder_activation_type = "tanh" if not use_final_dec_activation else config_dict["decoder_activation_type"]

    model = Autoencoder(
        num_layers=num_layers,
        image_channel=image_channel,
        min_channel=min_channel,
        max_channel=max_channel,
        latent_channel=latent_channel,
        hidden_activation_type=hidden_activation_type,
        use_final_enc_activation=use_final_enc_activation,
        encoder_activation_type=encoder_activation_type,
        use_final_dec_activation=use_final_dec_activation,
        decoder_activation_type=decoder_activation_type)

    model = model.to(device)
    model_optim = torch.optim.Adam(
        model.parameters(),
        lr=model_lr,
        betas=(0.5, 0.999))

    # Load saved model Parameters where possible.
    if model_path is not None:
        model_status, model_dict = load_model(model_path)
        if not model_status:
            raise Exception("An error occured while loading model checkpoint!")

        model.custom_load_state_dict(model_dict["model"])
        if load_optim:
            model_optim.load_state_dict(model_dict["model_optimizer"])

        # Update learning rate in case of changes.
        for model_optim_ in model_optim.param_groups:
            model_optim_["lr"] = model_lr

    # Image Dataset.
    image_dataset = ImageDataset(
        dataset_path=dataset_path,
        return_filepaths=False)
    image_dataloader = torch.utils.data.DataLoader(
        image_dataset,
        batch_size=batch_size,
        num_workers=4,
        shuffle=True)

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

    model_params_size = sum(param.numel() for param in model.parameters())

    logging.info(f"{project_name}")
    logging.info(f"Output Dir: {out_dir}")
    logging.info(f"Model size: {model_params_size:,}")
    logging.info("#" * 100)
    logging.info(f"Autoencoder Parameters.")
    logging.info(f"Num Layers: {num_layers:,}")
    logging.info(f"Image Channel: {image_channel:,}")
    logging.info(f"Min Channel: {min_channel:,}")
    logging.info(f"Max Channel: {max_channel:,}")
    logging.info(f"Latent Channel: {latent_channel:,}")
    logging.info(f"Hidden activation type: {hidden_activation_type}")
    if use_final_enc_activation:
        logging.info(f"Encoder activation type: {encoder_activation_type}")
    if use_final_dec_activation:
        logging.info(f"Decoder activation type: {decoder_activation_type}")
    logging.info("#" * 100)
    logging.info(f"Training Parameters.")
    logging.info(f"Max Epoch: {max_epoch:,}")
    logging.info(f"Batch Size: {batch_size:,}")
    logging.info(f"Model LR Update size: {lr_update_step:,}")
    logging.info(f"Model Checkpoint step: {model_checkpoint_step:,}")
    logging.info("#" * 100)

    for _ in range(curr_epoch, max_epoch):
        total_recon_loss = 0

        iteration_count = 0

        for index, image in enumerate(image_dataloader):
            image = image.to(device)

            iteration_count += 1

            # Synthesizer training.
            model.train()

            model_optim.zero_grad()

            recon = model(image)

            recon_loss = F.mse_loss(
                recon,
                image)

            if torch.isnan(recon_loss):
                raise Exception("NaN encountered during training")

            recon_loss.backward()

            model_optim.step()

            total_recon_loss += recon_loss.item()

            if global_steps % lr_update_step == 0 and global_steps > 0:
                for model_optim_ in model_optim.param_groups:
                    model_optim_['lr'] = model_optim_['lr'] * 0.5

            # Checkpoint and Plot Images.
            if global_steps % model_checkpoint_step == 0 and global_steps >= 0:
                # Save diffusion models, and optimizers.
                model_dict = {
                    "num_layers": num_layers,
                    "image_channel": image_channel,
                    "min_channel": min_channel,
                    "max_channel": max_channel,
                    "latent_channel": latent_channel,
                    "hidden_activation_type": hidden_activation_type,
                    "use_final_enc_activation": use_final_enc_activation,
                    "encoder_activation_type": encoder_activation_type,
                    "use_final_dec_activation": use_final_dec_activation,
                    "decoder_activation_type": decoder_activation_type,
                    "model": model.state_dict(),
                    "model_optimizer": model_optim.state_dict()}

                save_status = save_model(
                    model_dict=model_dict,
                    dest_path=out_dir,
                    file_name=f"model_{global_steps}.pt",
                    logging=logging.info)
                if save_status is True:
                    logging.info("Successfully saved model.")
                else:
                    logging.info("Error occured saving model.")
                
                # Save Images.
                _ = save_images(
                    images=image,
                    file_name=f"ground_truth_{global_steps}",
                    dest_path=out_dir,
                    logging=logging.info)
                _ = save_images(
                    images=recon,
                    file_name=f"recon_{global_steps}",
                    dest_path=out_dir,
                    logging=logging.info)

            temp_avg_recon_loss = total_recon_loss / iteration_count
            message = "Cum. Steps: {:,} | Steps: {:,} / {:,} | L.R.: {:.8f} | Recon Loss: {:.5f}".format(
                global_steps + 1,
                index + 1,
                len(image_dataloader),
                model_optim.param_groups[0]['lr'],
                temp_avg_recon_loss)
            logging.info(message)

            global_steps += 1

if __name__ == "__main__":
    main()
