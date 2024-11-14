import os
import json
import pathlib
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F

# Models.
from models.Codebook import Codebook
from models.FC_Decoder import FC_Decoder
from models.Transformer import Transformer

# Utility functions.
from utils.model_utils import load_model
from utils.image_utils import save_images

def main():
    parser = argparse.ArgumentParser(
        description=f"Generate Images.")

    parser.add_argument(
        "--device",
        help="Which hardware device will model run on.",
        choices=['cpu', 'cuda'],
        type=str,
        default="cpu")
    parser.add_argument(
        "--decoder-path",
        help="File path to pre-trained decoder model.",
        required=True,
        type=pathlib.Path)
    parser.add_argument(
        "--num-images",
        help="Num of images to generate.",
        type=int,
        default=25)
    parser.add_argument(
        "--seed",
        help="Seed value.",
        type=int,
        default=None)
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

    device = args["device"]  # Device to run model on.
    seed = args["seed"]
    decoder_path = args["decoder_path"]  #  Pre-trained decoder model path.
    num_images = args["num_images"]  # Number of images to generate.
    out_dir = args["out_dir"]  # Output Directory.
    try:
        os.makedirs(out_dir, exist_ok=True)
    except Exception as e:
        raise e

    if seed is not None:
        torch.manual_seed(seed)

    # Load and Parse config JSON.
    config_json = args["config_path"]
    with open(config_json, 'r') as json_file:
        json_data = json_file.read()
    config_dict = json.loads(json_data)

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

    for index, data in config_dict.items():
        print(f"Model: {int(index):,}")

        # Initial Index of sliding window for slice index.
        start_index = 0

        # Current Model params.
        model_path = data["model_path"]
        temperature = data["temperature"]
        lr_codebook_path = data["lr_codebook_path"]
        hr_codebook_path = data["hr_codebook_path"]
        num_beam = data["num_beam"]  # Set to 1 to generate 1 token at a time.
        beam_width = data["beam_width"]  # Needs to be divisible by total_Seq to work.

        # Load Low-Res Codebook and it's parameters (If Base model).
        if lr_codebook_path is not None:
            lr_codebook_status, lr_codebook_dict = load_model(lr_codebook_path)
            if not lr_codebook_status:
                raise Exception("An error occured while loading codebook checkpoint!")

            lr_codebook_num_embeddings = lr_codebook_dict["num_embeddings"]
            lr_codebook_patch_dim = lr_codebook_dict["patch_dim"]
            lr_codebook_image_dim = lr_codebook_dict["image_dim"]
            lr_codebook_image_C = lr_codebook_dict["image_C"]
            lr_codebook_neighbourhood_range = lr_codebook_dict["neighbourhood_range"]

            # Initialize High-Resolution Codebook.
            lr_codebook = Codebook(
                patch_dim=lr_codebook_patch_dim,
                image_dim=lr_codebook_image_dim,
                image_channel=lr_codebook_image_C,
                num_embeddings=lr_codebook_num_embeddings,
                init_neighbour_range=lr_codebook_neighbourhood_range)
            lr_codebook.custom_load_state_dict(lr_codebook_dict["checkpoint"])
            lr_codebook = lr_codebook.to(device)

        # Load High-Res Codebook and it's parameters.
        hr_codebook_status, hr_codebook_dict = load_model(hr_codebook_path)
        if not hr_codebook_status:
            raise Exception("An error occured while loading codebook checkpoint!")

        hr_codebook_patch_dim = hr_codebook_dict["patch_dim"]
        hr_codebook_num_embeddings = hr_codebook_dict["num_embeddings"]
        hr_codebook_image_dim = hr_codebook_dict["image_dim"]
        hr_codebook_image_C = hr_codebook_dict["image_C"]
        hr_codebook_neighbourhood_range = hr_codebook_dict["neighbourhood_range"]

        # Initialize High-Resolution Codebook.
        hr_codebook = Codebook(
            patch_dim=hr_codebook_patch_dim,
            image_dim=hr_codebook_image_dim,
            image_channel=hr_codebook_image_C,
            num_embeddings=hr_codebook_num_embeddings,
            init_neighbour_range=hr_codebook_neighbourhood_range)
        hr_codebook.custom_load_state_dict(hr_codebook_dict["checkpoint"])
        hr_codebook = hr_codebook.to(device)

        # Total number of sequence passed in Decoder.
        img_H, img_W = hr_codebook_image_dim
        patch_H, patch_W = hr_codebook_patch_dim
        total_Seq = (
            img_H // patch_H) * (
                img_W // patch_W)
        
        # Ensure beam_width is valid to avoid generate right number of token.
        # TODO: Implement a better way to avoid this check.
        if total_Seq % beam_width != 0:
            raise Exception("Invalid value for beam_width!")

        # Load saved model Parameters.
        model_status, model_dict = load_model(model_path)
        if not model_status:
            raise Exception("An error occured while loading model checkpoint!")

        train_base_model = model_dict["train_base_model"]
        use_sliding_window = model_dict["use_sliding_window"]
        sliding_window = model_dict["sliding_window"]
        num_enc_layers = model_dict["num_enc_layers"]
        num_dec_layers = model_dict["num_dec_layers"]
        num_enc_embedding = model_dict["num_enc_embedding"]
        num_dec_embedding = model_dict["num_dec_embedding"]
        self_attn_heads = model_dict["self_attn_heads"]
        cross_attn_heads = model_dict["cross_attn_heads"]
        transformer_in_dim = model_dict["transformer_in_dim"]
        transformer_out_dim = model_dict["transformer_out_dim"]
        transformer_hidden_dim = model_dict["transformer_hidden_dim"]
        hidden_activation = model_dict["hidden_activation"]

        # Transformer Model.
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

        model.custom_load_state_dict(model_dict["model"])

        model = model.to(device)

        with torch.no_grad():
            model.eval()

            if index == "0":
                # Base Model.
                lr_input = None  # No Encoder layer input.

                # Set Base Codebook indices as the first token, conditional.
                # TODO: Pass indices as arguments, instead of random indices.
                hr_input = torch.randint(
                    low=0,
                    high=lr_codebook_num_embeddings,
                    size=(num_images,1),
                    device=device)  # (N,1)

                lr_codebook.eval()
                decoder_model.eval()
                lr_quant = lr_codebook.get_quantized_image(
                    indices=hr_input,
                    unpatchify_input=True)
                lr_recon_decoder = decoder_model(lr_quant)

                # Save Images.
                _ = save_images(
                    images=lr_recon_decoder,
                    file_name=f"recon_model_Cond",
                    dest_path=out_dir,
                    logging=print)
            else:
                # Conditional input from previous High-Res input.
                lr_input = hr_input

                # Set <start> tokens the first token.
                hr_input = torch.tensor(
                    [[hr_codebook_num_embeddings]],
                    device=device).repeat(num_images,1)  # (N,1)

            # Conditional information, where sliding window is used.
            pos_indices = None
            if use_sliding_window:
                # Initial Position indices, starts at 0.
                pos_indices = torch.zeros(
                    (num_images,1),
                    device=device)  # (N,1)

            _, curr_num_seq = hr_input.shape

            # TODO: Remove possibility of over/under generating tokens in cases of invalid beam_width.
            while curr_num_seq < total_Seq:
                # Temporary variables for storing best sequences and computations.
                best_hr_input = None
                best_combined_prob = None

                # Iterative test multiple sequence prediction and pick the best.
                for _ in range(num_beam):
                    total_combined_prob = 1.0

                    temp_index = start_index
                    temp_hr_input = hr_input
                    temp_pos_indices = pos_indices

                    # Iteratively generate a sequence of tokens.
                    for token_count in range(beam_width):
                        # Emulate sliding window for entire data inside beam search.
                        _, temp_Seq = temp_hr_input.shape

                        # Update sliding window indices and data once filled up.
                        if use_sliding_window:
                            _, temp_Seq = temp_hr_input.shape
                            if temp_Seq >= sliding_window:
                                temp_index = temp_index + 1
                                temp_pos_indices = temp_pos_indices[:, 1:]

                        temp_hr_window = temp_hr_input[:,temp_index:]

                        out_seq = model(
                            x_dec=temp_hr_window,
                            x_enc=lr_input,
                            pos_cond=temp_pos_indices)

                        # Pick the last sequence.
                        out_seq = out_seq[:, -1, :]  # (N, Class)

                        probs = F.softmax(out_seq / temperature, dim=1)  # (N, Class)

                        # Remove <end> token prediction from consideration.
                        probs[:, hr_codebook_num_embeddings] = 0.0

                        # Pick most likely token for next generation for each Token Sequence (Seq).
                        next_token = torch.multinomial(probs, 1)

                        # Accumulate probs to be used in picking best tokens.
                        # P(Y) = P(y_1) * P(y_2|y_1) * P(y_3|y_1,y_2) * ... * P(y_t|y_1,y_2,...,y_t-1)
                        next_token_probs = probs[
                            torch.arange(num_images),
                            next_token.squeeze(dim=1)]  # (N,)
                        total_combined_prob = total_combined_prob * next_token_probs  # (N,)

                        if index == "0":
                            # Shift indices to appropriate range of values.
                            next_token = next_token + lr_codebook_num_embeddings

                        # Append next token to original input.
                        temp_hr_input = torch.cat(
                            (temp_hr_input,next_token),
                            dim=1)

                        if use_sliding_window:
                            # Append next position indices.
                            temp_indices = torch.tensor(
                                [[curr_num_seq + token_count + 1]],
                                device=device).repeat(num_images, 1)
                            temp_pos_indices = torch.cat(
                                (temp_pos_indices, temp_indices),
                                dim=1)

                    # Only keeps the best 1 sequence.
                    if best_combined_prob is None:
                        best_hr_input = temp_hr_input
                        best_combined_prob = total_combined_prob
                    else:
                        mask_prob = (best_combined_prob >= total_combined_prob).float()  # (N,)
                        best_combined_prob = (
                            mask_prob * best_combined_prob) + (
                                (1 - mask_prob) * total_combined_prob)

                        mask_seq = mask_prob[:, None]  # (N,1)
                        best_hr_input = (
                            mask_seq * best_hr_input) + (
                                (1 - mask_seq) * temp_hr_input)  # (N, Seq)

                start_index = temp_index

                hr_input = best_hr_input.long()
                if use_sliding_window:
                    pos_indices = temp_pos_indices.long()

                _, curr_num_seq = hr_input.shape

                print(f"{curr_num_seq - 1:,} / {total_Seq:,}")

            hr_input = hr_input[:,1:]  # Skip first token.
            if index == "0":
                # Revert to proper index.
                hr_input = hr_input - lr_codebook_num_embeddings

            hr_codebook.eval()
            decoder_model.eval()
            hr_quant = hr_codebook.get_quantized_image(
                indices=hr_input,
                unpatchify_input=True)
            hr_recon_decoder = decoder_model(hr_quant)

            # Save Images.
            _ = save_images(
                images=hr_recon_decoder,
                file_name=f"recon_model_{index}",
                dest_path=out_dir,
                logging=print)

            del model
            if use_sliding_window:
                del pos_indices
            torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
