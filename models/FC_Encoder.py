import torch
import torch.nn as nn

from .layers import (
    ConvLayer,
    DownsampleConvLayer)


"""
Fully Convolution Encoder Architecture.
"""
class FC_Encoder(nn.Module):
    def __init__(
            self,
            num_layers=2,
            image_channel=3,
            min_channel=128,
            max_channel=512,
            latent_channel=2,
            hidden_activation_type="silu",
            use_final_activation=True,
            final_activation_type="tanh"):
        super().__init__()

        curr_channel = min_channel

        self.fc_encoder_layer = nn.ModuleList()
        self.fc_encoder_layer.append(
            ConvLayer(
                in_channels=image_channel,
                out_channels=curr_channel,
                use_activation=True,
                activation_type=hidden_activation_type))

        for _ in range(num_layers):
            self.fc_encoder_layer.append(
                ConvLayer(
                    in_channels=curr_channel,
                    out_channels=curr_channel,
                    use_activation=True,
                    activation_type=hidden_activation_type)
            )

            next_channel = curr_channel * 2 if curr_channel * 2 < max_channel \
                else max_channel
            self.fc_encoder_layer.append(
                DownsampleConvLayer(
                    in_channels=curr_channel,
                    out_channels=next_channel,
                    activation_type=hidden_activation_type)
            )

            curr_channel = next_channel

        self.fc_encoder_layer.append(
            ConvLayer(
                in_channels=curr_channel,
                out_channels=latent_channel,
                use_activation=use_final_activation,
                activation_type=final_activation_type))

    def custom_load_state_dict(self, state_dict, ignore_msgs=False):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            # HACK: Load only encoder layers from autoencoder layers when passed.
            name = name.replace("fc_encoder.fc_encoder_layer", "fc_encoder_layer")

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

    def forward(self, x):
        for fc_enc_layer in self.fc_encoder_layer:
            x = fc_enc_layer(x)

        return x
