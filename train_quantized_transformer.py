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

from models.layers import (
    patchify,
    unpatchify)
from models.Codebook import Codebook
from models.FC_Decoder import FC_Decoder
from models.Transformer import Transformer

from dataset_loader.feature_map_dataset import FeatureMapDataset

# Utility functions.
from utils.model_utils import (
    save_model,
    load_model)
from utils.image_utils import save_images

def restricted_float(x):
    try:
        x = float(x)
    except ValueError:
        raise argparse.ArgumentTypeError("%r not a floating-point literal" % (x,))

    if x < 0.1:
        raise argparse.ArgumentTypeError("%r not in range > 0.1"%(x,))
    return x

def main():
    project_name = "Quantized Transformer"

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
        help="File path to feature map dataset json file.",
        required=True,
        type=pathlib.Path)
    parser.add_argument(
        "--train-base-model",
        action='store_true',
        help="Train Base Model, Decoder-only.")
    parser.add_argument(
        "--decoder-path",
        help="File path to pre-trained decoder model.",
        required=True,
        type=pathlib.Path)
    parser.add_argument(
        "--lr-codebook-path",
        help="File path to saved Low-Res codebook.",
        required=True,
        type=pathlib.Path)
    parser.add_argument(
        "--hr-codebook-path",
        help="File path to saved High-Res codebook.",
        required=True,
        type=pathlib.Path)
    parser.add_argument(
        "--model-path",
        help="File path to saved model checkpoint.",
        default=None,
        required=False,
        type=pathlib.Path)
    parser.add_argument(
        "--test-num-sample",
        help="Num samples for testing dataset.",
        type=int,
        default=25)
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
        "--temperature",
        help="Temperature for softmax sampling.",
        type=restricted_float,
        default=1.0)
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

    """
    Lower Temperature (T < 1):
    Prioritizes the most probable next token and effectively reduces
    randomness in the generated token.

    Higher Temperature (T > 1):
    Less probable tokens become more likely to be chosen, therefore more
    diversity in generated token.

    Can't be 0, OBVIOUSLY!!
    """
    temperature = args["temperature"]

    device = args["device"]  # Device to run model on.
    test_num_sample = args["test_num_sample"]
    train_base_model = args["train_base_model"]  # Train Decoder-only Transformer.
    decoder_path = args["decoder_path"]
    lr_codebook_path = args["lr_codebook_path"]  # File path to Low-Res codebook.
    hr_codebook_path = args["hr_codebook_path"]  # File path to High-Res codebook.
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

    # Pre-trained Decoder model.
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

    # Low - Resolution Codebook.
    lr_codebook_status, lr_codebook_dict = load_model(lr_codebook_path)
    if not lr_codebook_status:
        raise Exception("An error occured while loading Low-Resolution codebook checkpoint!")

    lr_patch_dim = lr_codebook_dict["patch_dim"]
    lr_num_embeddings = lr_codebook_dict["num_embeddings"]
    lr_image_dim = lr_codebook_dict["image_dim"]
    lr_image_C = lr_codebook_dict["image_C"]
    lr_neighbourhood_range = lr_codebook_dict["neighbourhood_range"]

    # Initialize Low-Resolution Codebook.
    lr_codebook = Codebook(
        patch_dim=lr_patch_dim,
        image_dim=lr_image_dim,
        image_channel=lr_image_C,
        num_embeddings=lr_num_embeddings,
        init_neighbour_range=lr_neighbourhood_range).to(device)
    lr_codebook.custom_load_state_dict(lr_codebook_dict["checkpoint"])

    # High - Resolution Codebook.
    hr_codebook_status, hr_codebook_dict = load_model(hr_codebook_path)
    if not hr_codebook_status:
        raise Exception("An error occured while loading Low-Resolution codebook checkpoint!")

    hr_patch_dim = hr_codebook_dict["patch_dim"]
    hr_num_embeddings = hr_codebook_dict["num_embeddings"]
    hr_image_dim = hr_codebook_dict["image_dim"]
    hr_image_C = hr_codebook_dict["image_C"]
    hr_neighbourhood_range = hr_codebook_dict["neighbourhood_range"]

    # Total number of sequence passed in Decoder.
    img_H, img_W = hr_image_dim
    hr_patch_H, hr_patch_W = hr_patch_dim
    total_hr_Seq = (
        img_H // hr_patch_H) * (
            img_W // hr_patch_W)

    # Initialize Low-Resolution Codebook.
    hr_codebook = Codebook(
        patch_dim=hr_patch_dim,
        image_dim=hr_image_dim,
        image_channel=hr_image_C,
        num_embeddings=hr_num_embeddings,
        init_neighbour_range=hr_neighbourhood_range).to(device)
    hr_codebook.custom_load_state_dict(hr_codebook_dict["checkpoint"])

    # Quantized Transformer Model Params.
    model_lr = config_dict["model_lr"]

    if train_base_model:
        # No Encoder Layer(s) Params.
        num_enc_layers = None
        num_enc_embedding = None

        # No Cross-Attention.
        cross_attn_heads = None

        # Combine LR codebook and HR codebook embeddings.
        num_dec_embedding = lr_num_embeddings + hr_num_embeddings
    else:
        # Encoder Layer(s) Params.
        num_enc_embedding = lr_num_embeddings
        num_enc_layers = config_dict["num_enc_layers"]

        # Number of Cross-Attention heads.
        cross_attn_heads = config_dict["cross_attn_heads"]

        # HR Codebook embeddings for Decoder.
        num_dec_embedding = hr_num_embeddings + 1  # Includes <start> token.

    # Number of Decoder layers
    num_dec_layers = config_dict["num_dec_layers"]

    # Sliding Window Params.
    use_sliding_window = config_dict["use_sliding_window"]
    if use_sliding_window:
        sliding_window = config_dict["sliding_window"]
    else:
        sliding_window = None

    # Number of Self-Attention heads.
    self_attn_heads = config_dict["self_attn_heads"]

    # Transformer model Params.
    transformer_in_dim = config_dict["in_dim"]
    transformer_out_dim = hr_num_embeddings + 1  # Includes <end> token.
    transformer_hidden_dim = config_dict["hidden_dim"]
    hidden_activation = config_dict["hidden_activation"]

    # Transformer model, to be trained.
    model = Transformer(
        use_encoder=not train_base_model,
        use_pos_cond=use_sliding_window,
        num_enc_layers=num_enc_layers,
        num_dec_layers=num_dec_layers,
        num_enc_embedding=num_enc_embedding,
        num_dec_embedding=num_dec_embedding,
        self_attn_heads=self_attn_heads,
        cross_attn_heads=cross_attn_heads,
        transformer_in_dim=transformer_in_dim,
        transformer_out_dim=transformer_out_dim,
        transformer_hidden_dim=transformer_hidden_dim,
        hidden_activation=hidden_activation)

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
        else:
            # Update learning rate in case of changes.
            for model_optim_ in model_optim.param_groups:
                model_optim_["lr"] = model_lr

    # Cross Entropy Loss.
    ce_loss = nn.CrossEntropyLoss()

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
    test_feature_map_dataloader = torch.utils.data.DataLoader(
        feature_map_dataset,
        batch_size=test_num_sample,
        num_workers=2,
        shuffle=True)

    model_params_size = sum(param.numel() for param in model.parameters())

    logging.info(f"{project_name}")
    logging.info(f"Output Dir: {out_dir}")
    logging.info(f"Model size: {model_params_size:,}")
    logging.info("#" * 100)
    logging.info(f"Decoder Parameters.")
    logging.info(f"Num Layers: {dec_num_layers:,}")
    logging.info(f"Image Channel: {dec_image_channel:,}")
    logging.info(f"Min Channel: {dec_min_channel:,}")
    logging.info(f"Max Channel: {dec_max_channel:,}")
    logging.info(f"Latent Channel: {dec_latent_channel:,}")
    logging.info(f"Hidden activation type: {dec_hidden_activation_type}")
    logging.info(f"Decoder activation type: {dec_final_activation_type}")
    logging.info("#" * 100)
    logging.info(f"Codebook Parameters.")
    logging.info(f"Low Res Patch size: {lr_patch_dim}")
    logging.info(f"Low Res Num Embeddings: {lr_num_embeddings:,}")
    logging.info(f"High Res Patch size: {hr_patch_dim}")
    logging.info(f"High Res Num Embeddings: {hr_num_embeddings:,}")
    logging.info("#" * 100)
    logging.info(f"Transformer Parameters.")
    if use_sliding_window:
        logging.info(f"Sliding Window: {sliding_window:,}")
    logging.info(f"Num Encoder Embedding: {num_enc_embedding}")
    logging.info(f"Num Encoder Layers: {num_enc_layers}")
    logging.info(f"Num Decoder Embedding: {num_dec_embedding:,}")
    logging.info(f"Num Decoder Layers: {num_dec_layers:,}")
    logging.info(f"Self Attention Heads: {self_attn_heads:,}")
    logging.info(f"Cross Attention Heads: {cross_attn_heads}")
    logging.info(f"In Dim: {transformer_in_dim:,}")
    logging.info(f"Out Dim: {transformer_out_dim:,}")
    logging.info(f"Hidden Dim: {transformer_hidden_dim:,}")
    logging.info(f"Hidden activation: {hidden_activation}")
    logging.info("#" * 100)
    logging.info(f"Training Parameters.")
    logging.info(f"Max Epoch: {max_epoch:,}")
    logging.info(f"Batch Size: {batch_size:,}")
    logging.info(f"Model LR Update size: {lr_update_step:,}")
    logging.info(f"Model Checkpoint step: {model_checkpoint_step:,}")
    logging.info("#" * 100)
    logging.info(f"Sampling Parameters.")
    logging.info(f"Temperature: {temperature:,}")
    logging.info("#" * 100)

    for epoch in range(curr_epoch, max_epoch):
        total_loss = 0
        iteration_count = 0

        for index, feature_map in enumerate(feature_map_dataloader):
            iteration_count = iteration_count + 1

            feature_map = feature_map.to(device)

            N, _, _, _ = feature_map.shape

            # Indices from Low-Resolution Codebook.
            lr_codebook.eval()
            lr_indices = lr_codebook.get_patches_bmu(
                feature_map,
                reshape=True)  # (N, lr_Seq)

            # Indices from High-Resolution Codebook.
            hr_codebook.eval()
            hr_indices = hr_codebook.get_patches_bmu(
                feature_map,
                reshape=True)  # (N, hr_Seq)

            if train_base_model:
                # Shift High-Resolution indices to account for combined Embeddings.
                hr_indices_shifted = hr_indices + lr_num_embeddings

                # Combined HR and LR indices, with LR acting like <start> token.
                hr_input = torch.cat(
                    (lr_indices,hr_indices_shifted),
                    dim=1)

                # No Low-Res indices.
                lr_input = None
            else:
                # Generate <start> token to be added at the beginning.
                start_tensor = torch.tensor(
                    [[hr_num_embeddings]],
                    device=device).repeat(N, 1)
                hr_input = torch.cat(
                    (start_tensor,hr_indices),
                    dim=1)

                # Low-Resolution input for Encoder layer.
                lr_input = lr_indices.to(device)

            hr_input = hr_input.to(device)

            # Generate <end> token to be added at the end.
            end_tensor = torch.tensor(
                [[hr_num_embeddings]],
                device=device).repeat(N, 1)
            hr_target = torch.cat(
                (hr_indices, end_tensor),
                dim=1)
            hr_target = hr_target.to(device)

            # In cases where HR input is too large for memory, use sliding window.
            sliding_window_indices = None
            if use_sliding_window:
                # Generate all the possible values of the sliding window.
                hr_input_unfold = hr_input.unfold(
                    dimension=1,
                    size=sliding_window,
                    step=1)
                hr_target_unfold = hr_target.unfold(
                    dimension=1,
                    size=sliding_window,
                    step=1)

                _, num_sliding_windows, _ = hr_input_unfold.shape

                # Randomly pick a section of the data using a sliding window. 
                rand_indices = torch.randint(
                    low=0,
                    high=num_sliding_windows,
                    size=(N,))
                hr_input = hr_input_unfold[torch.arange(N), rand_indices, :]  # (N, D)
                hr_target = hr_target_unfold[torch.arange(N), rand_indices, :]  # (N, D)

                # Indices of sliding window to be used as conditional information
                sliding_window_indices = rand_indices.unsqueeze(dim=1) +\
                 torch.arange(sliding_window).unsqueeze(dim=0)  # (N, Window)

                sliding_window_indices = sliding_window_indices.to(device)

            model.train()

            model_optim.zero_grad()

            classification_out = model(
                x_dec=hr_input,
                x_enc=lr_input,
                pos_cond=sliding_window_indices)

            # Flatten dimensions for loss.
            _, Seq, C = classification_out.shape            
            out_seq_flat = classification_out.view(N*Seq, C)  # (N*Seq, Class)
            target_seq_flat = hr_target.flatten()  # (N*Seq,)

            loss = ce_loss(
                out_seq_flat,
                target_seq_flat)
            if torch.isnan(loss):
                raise Exception("NaN encountered during training.")

            loss.backward()

            model_optim.step()

            total_loss = total_loss + loss.item()

            if global_steps % lr_update_step == 0 and global_steps > 0:
                for model_optim_ in model_optim.param_groups:
                    model_optim_['lr'] = model_optim_['lr'] * 0.5

            # Checkpoint models and synthesis images.
            if global_steps % model_checkpoint_step == 0 and global_steps >= 0:
                # Save models, and optimizers.
                model_dict = {
                    "train_base_model": train_base_model,
                    "use_sliding_window": use_sliding_window,
                    "sliding_window": sliding_window,
                    "num_enc_embedding": num_enc_embedding,
                    "num_dec_embedding": num_dec_embedding,
                    "num_enc_layers": num_enc_layers,
                    "num_dec_layers": num_dec_layers,
                    "self_attn_heads": self_attn_heads,
                    "cross_attn_heads": cross_attn_heads,
                    "transformer_in_dim": transformer_in_dim,
                    "transformer_out_dim": transformer_out_dim,
                    "transformer_hidden_dim": transformer_hidden_dim,
                    "hidden_activation": hidden_activation,
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

                # Autoregressively generate image one token at a time.
                model.eval()
                lr_codebook.eval()
                hr_codebook.eval()
                decoder_model.eval()

                with torch.no_grad():
                    test_feature_map = next(iter(test_feature_map_dataloader))
                    test_feature_map = test_feature_map.to(device)

                    latent_decoder = decoder_model(test_feature_map)

                    test_hr_quant = hr_codebook(test_feature_map)
                    test_lr_quant = lr_codebook(test_feature_map)

                    hr_decoder = decoder_model(test_hr_quant)
                    lr_decoder = decoder_model(test_lr_quant)

                    # Save Images.
                    _ = save_images(
                        images=latent_decoder,
                        file_name=f"ground_truth_{global_steps}",
                        dest_path=out_dir,
                        logging=logging.info)
                    _ = save_images(
                        images=lr_decoder,
                        file_name=f"low_res_cond_{global_steps}",
                        dest_path=out_dir,
                        logging=logging.info)
                    _ = save_images(
                        images=hr_decoder,
                        file_name=f"high_res_example_{global_steps}",
                        dest_path=out_dir,
                        logging=logging.info)

                    start_index = 0

                    if train_base_model:
                        # No Encoder layer input.
                        test_lr_input = None

                        # Set low-resolution as the first token.
                        test_hr_input = lr_codebook.get_patches_bmu(
                            test_feature_map,
                            reshape=True)  # (N, lr_Seq)
                    else:
                        # Conditional input for Encoder layer.
                        test_lr_input = lr_codebook.get_patches_bmu(
                            test_feature_map,
                            reshape=True)  # (N, lr_Seq)

                        # Set <start> tokenas the first token.
                        test_hr_input = torch.tensor(
                            [[hr_num_embeddings]],
                            device=device).repeat(test_num_sample,1)  # (test_N,1)

                    # Conditional information, where sliding window is used.
                    test_pos_indices = None
                    if use_sliding_window:
                        # Initial Position indices, starts at 0.
                        test_pos_indices = torch.zeros(
                            (test_num_sample,1),
                            device=device)  # (test_N,1)

                    for step in range(total_hr_Seq):
                        print(f"{step + 1:,} / {total_hr_Seq:,}")

                        # Update sliding window indices and data once filled up.
                        if use_sliding_window:
                            _, temp_Seq = test_hr_input.shape
                            if temp_Seq >= sliding_window:
                                start_index = start_index + 1
                                test_pos_indices = test_pos_indices[:, 1:]

                        test_hr_window = test_hr_input[:,start_index:]

                        out_seq = model(
                            x_dec=test_hr_window,
                            x_enc=test_lr_input,
                            pos_cond=test_pos_indices)

                        # Pick the last sequence.
                        out_seq = out_seq[:, -1, :]  # (N, Class)

                        probs = F.softmax(out_seq / temperature, dim=1)  # (N, Class)

                        # Pick most likely token for next generation for each Token Sequence (Seq).
                        next_token = torch.multinomial(probs, 1)

                        # HACK: Replace end token prediction(s) with default 0 index.
                        next_token[next_token==hr_num_embeddings] = 0

                        if train_base_model:
                            # Shift indices to appropriate range of values.
                            next_token = next_token + lr_num_embeddings

                        # Append next token to original input.
                        test_hr_input = torch.cat(
                            (test_hr_input,next_token),
                            dim=1)

                        if use_sliding_window:
                            # Append next position indices.
                            temp_indices = torch.tensor(
                                [[step + 1]],
                                device=device).repeat(test_num_sample, 1)
                            test_pos_indices = torch.cat(
                                (test_pos_indices, temp_indices),
                                dim=1)

                    test_hr_input = test_hr_input[:,1:]  # Skip first token.
                    if train_base_model:
                        test_hr_input = test_hr_input - lr_num_embeddings  # Revert to proper index.
                        test_hr_input[test_hr_input==hr_num_embeddings] = lr_num_embeddings  # In cases where <end> token is predicted.
                    else:
                        test_hr_input[test_hr_input==hr_num_embeddings] = 0  # In cases where <end> token is predicted.

                    test_hr_quant = hr_codebook.get_quantized_image(
                        indices=test_hr_input,
                        unpatchify_input=True)
                    hr_recon_decoder = decoder_model(test_hr_quant)

                    # Save Images.
                    _ = save_images(
                        images=hr_recon_decoder,
                        file_name=f"high_res_recon_{global_steps}",
                        dest_path=out_dir,
                        logging=logging.info)

                    del test_hr_input
                    del test_pos_indices
                    torch.cuda.empty_cache()

            temp_avg_loss = total_loss / iteration_count
            message = "Cum. Steps: {:,} | Steps: {:,} / {:,} | L.R.: {:.8f} | Recon Loss: {:.5f}".format(
                global_steps + 1,
                index + 1,
                len(feature_map_dataloader),
                model_optim.param_groups[0]['lr'],
                temp_avg_loss)
            logging.info(message)

            global_steps += 1

if __name__ == "__main__":
    main()
