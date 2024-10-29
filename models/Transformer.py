import torch
import torch.nn as nn

from .layers import (
    LinearLayer,
    TransformerBlock,
    get_positional_embeddings)


"""
Transformer architecture: https://arxiv.org/abs/1706.03762
"""
class Transformer(nn.Module):
    def __init__(
            self,
            num_enc_embedding,
            num_dec_embedding,
            num_enc_layers,
            num_dec_layers,
            self_attn_heads,
            cross_attn_heads,
            transformer_in_dim,
            transformer_out_dim,
            transformer_hidden_dim,
            hidden_activation="silu"):
        super().__init__()

        self.enc_embedding = nn.Embedding(
            num_embeddings=num_enc_embedding,
            embedding_dim=transformer_in_dim)
        self.dec_embedding = nn.Embedding(
            num_embeddings=num_dec_embedding + 1,
            embedding_dim=transformer_in_dim)

        self.encoder_layers = nn.ModuleList()
        for _ in range(num_enc_layers):
            self.encoder_layers.append(
                TransformerBlock(
                    self_attn_heads=self_attn_heads,
                    in_dim=transformer_in_dim,
                    hidden_dim=transformer_hidden_dim,
                    use_cross_attn=False,
                    use_masked_attn=False,
                    activation_type=hidden_activation))

        self.decoder_layers = nn.ModuleList()
        for _ in range(num_dec_layers):
            self.decoder_layers.append(
                TransformerBlock(
                    self_attn_heads=self_attn_heads,
                    cross_attn_heads=cross_attn_heads,
                    in_dim=transformer_in_dim,
                    cond_dim=transformer_in_dim,
                    hidden_dim=transformer_hidden_dim,
                    use_cross_attn=True,
                    use_masked_attn=True,
                    activation_type=hidden_activation))

        self.classifier = nn.Sequential(
            LinearLayer(
                in_dim=transformer_in_dim,
                out_dim=transformer_hidden_dim,
                use_activation=True),
            LinearLayer(
                in_dim=transformer_hidden_dim,
                out_dim=transformer_out_dim + 1,
                use_activation=False))

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

    def forward(self, x, cond):
        """
        Encoder Half.
        """
        x_enc = self.enc_embedding(cond)

        _, enc_Seq, D = x_enc.shape
        enc_pos_index = torch.arange(
            start=1,
            end=enc_Seq + 1,
            device=cond.device)  # (enc_Seq,)
        enc_pos_emb = get_positional_embeddings(
            emb_dim=D,
            pos_index=enc_pos_index).unsqueeze(dim=0)  # (1, enc_Seq, D)

        # Adds Positional to the input.
        x_enc = x_enc + enc_pos_emb

        for encoder_layer in self.encoder_layers:
            x_enc = encoder_layer(x_enc)

        """
        Decoder Half.
        """
        x_dec = self.dec_embedding(x)

        _, dec_Seq, _ = x_dec.shape
        dec_pos_index = torch.arange(
            start=1,
            end=dec_Seq + 1,
            device=x.device)  # (enc_Seq,)
        dec_pos_emb = get_positional_embeddings(
            emb_dim=D,
            pos_index=dec_pos_index).unsqueeze(dim=0)  # (1, dec_Seq, D)

        # Adds Positional to the input.
        x_dec = x_dec + dec_pos_emb

        for decoder_layer in self.decoder_layers:
            x_dec = decoder_layer(
                x=x_dec,
                cross_cond=x_enc)

        x_class = self.classifier(x_dec)
        return x_class
