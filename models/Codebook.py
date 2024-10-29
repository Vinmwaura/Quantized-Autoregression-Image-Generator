import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import (
    patchify,
    unpatchify)


"""
Codebook, trained using a slightly tweaked Self Organizing Map (SOM) algorithm.
Encoder half of the pre-trained Autoencoder is used to compute the latent image
which is patchified. The patches are then used to compute the Codebook.
"""
class Codebook(nn.Module):
    def __init__(
            self,
            patch_dim=(2,2),
            image_dim=(32,32),
            image_channel=4,
            num_embeddings=512,
            init_neighbour_range=256):
        super().__init__()

        if init_neighbour_range > num_embeddings and init_neighbour_range < 1:
            raise Exception("Invalid value for init_neighbour_range.")
        self.neighbourhood_range = init_neighbour_range

        self.patch_dim = patch_dim  # (patch_H, patch_W).
        self.image_dim = image_dim  # (image_H, image_W).

        # Embedding dimension of each patches.
        patch_H, patch_W = self.patch_dim
        self.embedding_dim = image_channel * patch_H * patch_W

        self.num_embeddings = num_embeddings

        # Codebook.
        self.codebook = nn.Embedding(
            self.num_embeddings,
            self.embedding_dim)
        self.codebook.weight.data.uniform_(
            -1 / self.num_embeddings,
            1 / self.num_embeddings)

    def custom_load_state_dict(self, state_dict, ignore_msgs=False):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name not in own_state:
                if not ignore_msgs:
                    print(f"No Layer found: {name}, skipping")
                continue

            # Skip loading mismatched weights, in cases of weight changes.
            if (own_state[name].shape != param.data.shape):
                if not ignore_msgs:
                    print(f"Skipped: {name}")
                continue

            if isinstance(param, torch.nn.parameter.Parameter):
                # Backwards compatibility for serialized parameters
                param = param.data

            own_state[name].copy_(param)

    def decrease_neighbourhood(self, steps=1):
        if steps < 1:
            raise Exception("Invalid value for steps, should be > 1.")

        min_value = 1.0
        self.neighbourhood_range = min_value if self.neighbourhood_range <= 1\
            else self.neighbourhood_range - 1

    # Patchify Image input, and then get BMU indices.
    def get_patches_bmu(self, x, reshape=False):
        x_patches = patchify(
            image=x,
            patch_dim=self.patch_dim)  # (N, Seq, D)

        N, Seq, D = x_patches.shape
        x_flat_patches = x_patches.reshape(N*Seq, D)  # (N*Seq, D)

        # Computes the distance between inputs and embeddings.
        distances = torch.cdist(
            x_flat_patches,
            self.codebook.weight)  # (N*Seq, Num_embs)

        # Get the indices of the closest embedding vector.
        bmu_indices = torch.argmin(
            distances,
            dim=-1,
            keepdim=False)  # (N*Seq)

        if reshape:
            bmu_indices = bmu_indices.reshape(N, Seq)

        return bmu_indices

    # Compute discrete patches.
    def get_quantized_patches(
            self,
            x,
            use_gaussian=True):
        bmu_indices = self.get_patches_bmu(x)  # (N*Seq)

        N, _, _, _ = x.shape
        if use_gaussian:
            bmu_indices = bmu_indices.unsqueeze(dim=-1)  # (N*Seq, 1)

            embedding_indices = torch.arange(
                start=0,
                end=self.codebook.num_embeddings
            ).unsqueeze(dim=0).to(x.device)  # (1, Num_embs)

            # Computes the variance for the Gaussian to be close to 0 at neighbourhood range.
            variance = -(self.neighbourhood_range / (2 * math.log(0.1)))

            # Gaussian Function.
            neighbourhood_scale = torch.exp(
                -(
                    (embedding_indices - bmu_indices)**2 / (2 * variance)
                )
            )

            # Quantize the input.
            quantized_patches = torch.matmul(
                neighbourhood_scale,
                self.codebook.weight)  # (N*Seq, Num_embs) @ (Num_embs, D) => (N*Seq, D)
        else:
            quantized_patches = self.codebook(bmu_indices)  #  (N*Seq, D)

        quantized_patches = quantized_patches.view(N, -1, self.embedding_dim)  # (N, Seq, D)
        return quantized_patches

    # Compute discrete image using codebook indices.
    def get_quantized_image(
            self,
            indices,
            unpatchify_input=True):
        N, Seq = indices.shape
        flat_indices = indices.flatten()  # (N*Seq)
        quantized_flat_patches = self.codebook(flat_indices)  # (N*Seq, D)
        quantized_patches = quantized_flat_patches.view(
            N, Seq, self.embedding_dim)  # (N, Seq, D)

        if unpatchify_input:
            quantized_image = unpatchify(
                patches=quantized_patches,
                image_dim=self.image_dim,
                patch_dim=self.patch_dim)
            return quantized_image
        return quantized_patches

    def forward(self, x, use_gaussian=True):
        x_patchified_quant = self.get_quantized_patches(
            x,
            use_gaussian=use_gaussian)
        x_unpatchified = unpatchify(
            patches=x_patchified_quant,
            image_dim=self.image_dim,
            patch_dim=self.patch_dim)
        return x_unpatchified
