import torch
import torch.nn as nn

from .FC_Encoder import FC_Encoder
from .FC_Decoder import FC_Decoder


"""
Fully Convolutional Autoencoder Architecture.
"""
class Autoencoder(nn.Module):
    def __init__(
            self,
            num_layers=2,
            image_channel=3,
            min_channel=128,
            max_channel=512,
            latent_channel=2,
            hidden_activation_type="silu",
            use_final_enc_activation=True,
            encoder_activation_type="silu",
            use_final_dec_activation=True,
            decoder_activation_type="tanh"):
        super().__init__()

        self.fc_encoder = FC_Encoder(
            num_layers=num_layers,
            image_channel=image_channel,
            min_channel=min_channel,
            max_channel=max_channel,
            latent_channel=latent_channel,
            hidden_activation_type=hidden_activation_type,
            use_final_activation=use_final_enc_activation,
            final_activation_type=encoder_activation_type)
        self.fc_decoder = FC_Decoder(
            num_layers=num_layers,
            image_channel=image_channel,
            min_channel=min_channel,
            max_channel=max_channel,
            latent_channel=latent_channel,
            hidden_activation_type=hidden_activation_type,
            use_final_activation=use_final_dec_activation,
            final_activation_type=decoder_activation_type)

    def custom_load_state_dict(self, state_dict):
        own_state = self.state_dict()
        for name, param in state_dict.items():
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

    def get_latent(self, x):
        z = self.fc_encoder(x)
        return z

    def recon_image(self, z):
        recon = self.fc_decoder(z)
        return recon

    def forward(self, x):
        z = self.get_latent(x)
        recon = self.recon_image(z)
        return recon
