import math

import torch
import torch.nn as nn
import torch.nn.functional as F

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


