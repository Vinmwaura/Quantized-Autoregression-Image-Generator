import torch
import torch.nn as nn

from .layers import (
    ConvLayer,
    UpsampleConvLayer)


"""
Fully Convolution Decoder Architecture.
"""
class FC_Decoder(nn.Module):
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

        curr_channel = max_channel

        self.fc_decoder_layer = nn.ModuleList()
        self.fc_decoder_layer.append(
            nn.Sequential(
                ConvLayer(
                    in_channels=latent_channel,
                    out_channels=curr_channel,
                    use_activation=True,
                    activation_type=hidden_activation_type),
                ConvLayer(
                    in_channels=curr_channel,
                    out_channels=curr_channel,
                    use_activation=True,
                    activation_type=hidden_activation_type)))

        for _ in range(num_layers):
            self.fc_decoder_layer.append(
                ConvLayer(
                    in_channels=curr_channel,
                    out_channels=curr_channel,
                    use_activation=True,
                    activation_type=hidden_activation_type)
            )

            next_channel = curr_channel // 2 if curr_channel // 2 > min_channel \
                else min_channel
            self.fc_decoder_layer.append(
                UpsampleConvLayer(
                    in_channels=curr_channel,
                    out_channels=next_channel,
                    activation_type=hidden_activation_type)
            )

            curr_channel = next_channel

        self.fc_decoder_layer.append(
            ConvLayer(
                in_channels=curr_channel,
                out_channels=image_channel,
                use_activation=use_final_activation,
                activation_type=final_activation_type))

    def custom_load_state_dict(self, state_dict):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            # HACK: Load only decoder layers from autoencoder layers when passed.
            name = name.replace("fc_decoder.fc_decoder_layer", "fc_decoder_layer")

            # Only load decoder weights in cases of Autoencoder weights.
            if not "decoder" in name:
                print(f"Skipping: {name}")
                continue

            if name not in own_state:
                print(f"No Layer found: {name}, skipping")
                continue

            # Skip loading mismatched weights, in cases of weight changes.
            if (own_state[name].shape != param.data.shape):
                print(f"Skipped: {name}")
                continue

            if isinstance(param, torch.nn.parameter.Parameter):
                # Backwards compatibility for serialized parameters
                param = param.data
            own_state[name].copy_(param)

    def forward(self, x):
        for fc_dec_layer in self.fc_decoder_layer:
            x = fc_dec_layer(x)
        return x
