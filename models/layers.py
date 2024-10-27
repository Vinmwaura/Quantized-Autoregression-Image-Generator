import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# Function to break images into patches.
def patchify(image, patch_dim=(4,4)):
    patch_H, patch_W = patch_dim

    image_N, image_C, image_H, image_W = image.shape
    new_image_H = image_H // patch_H
    new_image_W = image_W // patch_W

    # Reshape the image to create patches.
    # (N, C, new_H, patch_height, new_W, patch_width).
    image_patches = image.reshape(
        image_N,
        image_C,
        new_image_H,
        patch_H,
        new_image_W,
        patch_W)

    # Swap the axes to get the patches in the desired order.
    # (N, new_H, new_W, C, patch_height, patch_width).
    image_patches = image_patches.permute(0, 2, 4, 1, 3, 5)

    # Flatten each patches to dimension d, and combine the patches together.
    image_patches = image_patches.reshape(
        image_N,
        new_image_H*new_image_W,
        image_C*patch_H*patch_W)  # (N, Seq, D)
    return image_patches

# Function to reassemble patches into images.
def unpatchify(
        patches,
        image_dim=(32,32),
        patch_dim=(4,4)):
    image_H, image_W = image_dim
    patch_H, patch_W = patch_dim

    image_N, _, image_D = patches.shape

    # Seq => new_height * new_width
    new_image_H = image_H // patch_H
    new_image_W = image_W // patch_W

    # D => C * patch_height * patch_width
    image_C = image_D // (patch_H * patch_W)

    # (N, new_H, new_W, C, patch_H, patch_W)
    patches = patches.reshape(
        image_N,
        new_image_H,
        new_image_W,
        image_C,
        patch_H,
        patch_W)

    # Swap the axes to get the patches in the desired order.
    # (N, C, new_H, patch_H, new_W, patch_W)
    patches = patches.permute(0, 3, 1, 4, 2, 5)

    image = patches.reshape(
        image_N,
        image_C,
        patch_H*new_image_H,
        patch_W*new_image_W)  # (N, C, H, W)
    return image

# List of activation to be used.
def get_activation(activation_type):
    activations_dict = nn.ModuleDict([
        ['silu', nn.SiLU()],  # aka swish.
        ['tanh', nn.Tanh()],
        ['sigmoid', nn.Sigmoid()],
    ])
    return activations_dict[activation_type]


# Convolution Layer.
class ConvLayer(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            use_activation=True,
            activation_type="silu"):
        super().__init__()

        conv_list = [
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding)
        ]
        if use_activation:
            conv_list.append(
                get_activation(activation_type))
        self.conv_layer = nn.Sequential(*conv_list)

    def forward(self, x):
        x = self.conv_layer(x)
        return x


# Convolution Upsample using ConvTranspose2d.
class UpsampleConvLayer(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            activation_type="silu"):
        super().__init__()

        self.conv_layer = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=4,
                stride=2,
                padding=1),
            get_activation(activation_type))

    def forward(self, x):
        x = self.conv_layer(x)
        return x


# Convolution Downsample using Conv2d.
class DownsampleConvLayer(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            activation_type="silu"):
        super().__init__()

        self.conv_layer = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=2,
                padding=1),
            get_activation(activation_type))

    def forward(self, x):
        x = self.conv_layer(x)
        return x


