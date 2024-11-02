import torch
import torch.nn as nn

from .layers import (
    LinearLayer,
    TransformerBlock,
    get_positional_embeddings)

"""
Vanilla Transformer + DiT architecture:
> https://arxiv.org/abs/1706.03762
> https://arxiv.org/abs/2212.09748
"""
class Transformer(nn.Module):
    def __init__(
            self,
            use_encoder=True,
            use_pos_cond=True,
            num_enc_layers=5,
            num_dec_layers=10,
            num_enc_embedding=512,
            num_dec_embedding=512,
            self_attn_heads=8,
            cross_attn_heads=8,
            transformer_in_dim=512,
            transformer_out_dim=512,
            transformer_hidden_dim=4096,
            hidden_activation="silu"):
        super().__init__()

        self.use_encoder = use_encoder
        self.use_pos_cond = use_pos_cond

        if self.use_encoder:
            self.enc_embedding = nn.Embedding(
                num_embeddings=num_enc_embedding,
                embedding_dim=transformer_in_dim)

            # Vanilla Transformer Architecture.
            self.encoder_layers = nn.ModuleList()
            for _ in range(num_enc_layers):
                self.encoder_layers.append(
                    TransformerBlock(
                        in_dim=transformer_in_dim,
                        hidden_dim=transformer_hidden_dim,
                        self_attn_heads=self_attn_heads,
                        use_cross_attn=False,
                        use_masked_attn=False,
                        use_adaln0=False,
                        use_scale_layer=False,
                        activation_type=hidden_activation))

        self.dec_embedding = nn.Embedding(
            num_embeddings=num_dec_embedding + 1,  # Includes <Start> Token.
            embedding_dim=transformer_in_dim)

        # Diffusion Transformer Architecture.
        self.decoder_layers = nn.ModuleList()
        for _ in range(num_dec_layers):
            self.decoder_layers.append(
                TransformerBlock(
                    in_dim=transformer_in_dim,
                    cond_dim=transformer_in_dim,
                    cross_cond_dim=transformer_in_dim,
                    hidden_dim=transformer_hidden_dim,
                    self_attn_heads=self_attn_heads,
                    cross_attn_heads=cross_attn_heads,
                    use_cross_attn=True,
                    use_masked_attn=True,
                    use_adaln0=self.use_pos_cond,
                    use_scale_layer=self.use_pos_cond,
                    activation_type=hidden_activation))

        # MLP for Positional conditioning.
        # If using sliding window, pass position of patches as condition.
        if self.use_pos_cond:
            self.pos_cond_layer = nn.Sequential(
                LinearLayer(
                    in_dim=transformer_in_dim,
                    out_dim=transformer_hidden_dim,
                    use_activation=True,
                    activation_type=hidden_activation),
                LinearLayer(
                    in_dim=transformer_hidden_dim,
                    out_dim=transformer_in_dim,
                    use_activation=False))

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

    def forward(self, x_dec, x_enc=None, pos_cond=None):
        if self.use_encoder:
            """
            Encoder Half.
            """
            x_enc = self.enc_embedding(x_enc)

            _, enc_Seq, enc_D = x_enc.shape
            enc_pos_index = torch.arange(
                start=1,
                end=enc_Seq + 1,
                device=x_enc.device)  # (enc_Seq,)
            enc_pos_emb = get_positional_embeddings(
                emb_dim=enc_D,
                pos_index=enc_pos_index).unsqueeze(dim=0)  # (1, enc_Seq, D)

            # Adds Positional to the input.
            x_enc = x_enc + enc_pos_emb

            for encoder_layer in self.encoder_layers:
                x_enc = encoder_layer(x_enc)

        """
        Decoder Half.
        """
        # Output Embedding.
        x_dec = self.dec_embedding(x_dec)  # (N, Seq, D)

        dec_N, dec_Seq, dec_D = x_dec.shape

        # Positional Encoding.
        dec_pos_index = torch.arange(
            start=1,
            end=dec_Seq + 1,
            device=x_dec.device)  # (enc_Seq,)
        dec_pos_emb = get_positional_embeddings(
            emb_dim=dec_D,
            pos_index=dec_pos_index).unsqueeze(dim=0)  # (1, dec_Seq, D)

        x_dec = x_dec + dec_pos_emb

        # Patch Position Conditioning.
        pos_cond_emb = None
        if self.use_pos_cond:
            pos_cond_flat = pos_cond.flatten()  # (N*Seq,)
            pos_cond_emb = get_positional_embeddings(
                emb_dim=dec_D,
                pos_index=pos_cond_flat).unsqueeze(dim=0)  # (N*Seq, D)
            pos_cond_emb = pos_cond_emb.reshape(dec_N, dec_Seq, dec_D)  # (N, Seq, D)
            pos_cond_emb = self.pos_cond_layer(pos_cond_emb)  # (N, Seq, D)

        for decoder_layer in self.decoder_layers:
            x_dec = decoder_layer(
                x=x_dec,
                cross_cond=x_enc,
                pos_cond=pos_cond_emb)

        x_class = self.classifier(x_dec)
        return x_class
