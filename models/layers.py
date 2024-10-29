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

# Function for generating sinusoidal positional embeddings.
def get_positional_embeddings(emb_dim, pos_index):
    half_dim = emb_dim // 2

    pos_emb = math.log(10_000) / (half_dim - 1)
    pos_emb = torch.exp(
        torch.arange(
            half_dim,
            dtype=torch.float32,
            device=pos_index.device) * -pos_emb)

    pos_emb = pos_index[:, None] * pos_emb[None, :]
    pos_emb = torch.cat((pos_emb.sin(), pos_emb.cos()), dim=1)

    return pos_emb


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


# Linear Layer.
class LinearLayer(nn.Module):
    def __init__(
            self,
            in_dim,
            out_dim,
            use_activation=True,
            activation_type="silu"):
        super().__init__()

        linear_list = [nn.Linear(in_dim, out_dim)]

        if use_activation:
            linear_list.append(
                get_activation(
                    activation_type=activation_type))

        self.linear_layer = nn.Sequential(*linear_list)

    def forward(self, x):
        x = self.linear_layer(x)
        return x


# Residual Linear Layer.
class ResidualLinearLayer(nn.Module):
    def __init__(
            self,
            in_dim=512,
            out_dim=512,
            skip_dim=512,
            activation_type="silu"):
        super().__init__()

        self.linear = LinearLayer(
            in_dim=in_dim,
            out_dim=out_dim,
            use_activation=False)

        if skip_dim != out_dim:
            self.skip_linear = LinearLayer(
                in_dim=skip_dim,
                out_dim=out_dim,
                use_activation=False)
        else:
            self.skip_linear = nn.Identity()

        self.activation = get_activation(
            activation_type=activation_type)

    def forward(self, x, x_skip):
        x = self.linear(x)
        x_skip = self.skip_linear(x_skip)

        # Skip connection.
        x = x + x_skip

        x = self.activation(x)
        return x


# Feedforward-Block: (Feedforward -> Norm -> Residual).
class FeedforwardBlock(nn.Module):
    def __init__(
            self,
            in_dim=512,
            hidden_dim=512,
            activation_type="silu"):
        super().__init__()

        # FeedForward Layer.
        self.feedforward = nn.Sequential(
            LinearLayer(
                in_dim=in_dim,
                out_dim=hidden_dim,
                use_activation=True,
                activation_type=activation_type),
            LinearLayer(
                in_dim=hidden_dim,
                out_dim=in_dim,
                use_activation=True,
                activation_type=activation_type))
        self.feedforward_norm = nn.LayerNorm(in_dim)
        self.feedforward_res = ResidualLinearLayer(
            in_dim=in_dim,
            out_dim=in_dim,
            skip_dim=in_dim,
            activation_type=activation_type)

    def forward(self, x):
        init_x = x

        x = self.feedforward(x)
        x = self.feedforward_norm(x)
        x = self.feedforward_res(
            x=x,
            x_skip=init_x)

        return x


# Attention Layer: https://arxiv.org/abs/1706.03762
class AttentionLayer(nn.Module):
    def __init__(
            self,
            heads=8,
            in_dim=512,
            cond_dim=512,
            hidden_dim=2_048,
            use_cross_attn=True,
            use_masked_attn=True,
            activation_type="silu"):
        super().__init__()

        self.heads = heads
        self.use_cross_attn = use_cross_attn
        self.use_masked_attn = use_masked_attn

        if not self.use_cross_attn:
            cond_dim = in_dim

        self.q_block = nn.Sequential(
            LinearLayer(
                in_dim=in_dim,
                out_dim=hidden_dim,
                use_activation=True,
                activation_type=activation_type),
            LinearLayer(
                in_dim=hidden_dim,
                out_dim=in_dim,
                use_activation=False))
        self.k_block = nn.Sequential(
            LinearLayer(
                in_dim=cond_dim,
                out_dim=hidden_dim,
                use_activation=True,
                activation_type=activation_type),
            LinearLayer(
                in_dim=hidden_dim,
                out_dim=in_dim,
                use_activation=False))
        self.v_block = nn.Sequential(
            LinearLayer(
                in_dim=cond_dim,
                out_dim=hidden_dim,
                use_activation=True,
                activation_type=activation_type),
            LinearLayer(
                in_dim=hidden_dim,
                out_dim=in_dim,
                use_activation=False))

    def forward(self, x, cross_cond=None):
        q = self.q_block(x)  # (N, Seq_q, D)

        if self.use_cross_attn:
            k = self.k_block(cross_cond)  # (N, Seq_k, D)
            v = self.v_block(cross_cond)  # (N, Seq_v, D)
        else:
            k = self.k_block(x)  # (N, Seq_k, D)
            v = self.v_block(x)  # (N, Seq_v, D)

        N_q, Seq_q, D = q.shape
        _, Seq_k, _ = k.shape
        _, Seq_v, _ = v.shape
        D_split = D // self.heads

        # Logically split into heads.
        # (N, H, Seq, D_split)
        q_head_split = q.reshape(
            N_q, Seq_q, self.heads, D_split).permute(0, 2, 1, 3)
        k_head_split = k.reshape(
            N_q, Seq_k, self.heads, D_split).permute(0, 2, 1, 3)
        v_head_split = v.reshape(
            N_q, Seq_v, self.heads, D_split).permute(0, 2, 1, 3)

        # (N, H, Seq_q, D_split),(N, H, Seq_k, D_split) => (N, H, Seq_q, Seq_k)
        qk_T = torch.einsum("nhqd,nhkd->nhqk", q_head_split, k_head_split)
        qk_T_normalized = qk_T / (D_split**0.5)  # (N, H, Seq_q, Seq_k)

        if self.use_masked_attn:
            _, Seq, _ = x.shape
            # (1, 1, Seq, Seq)
            mask = torch.ones((1, 1, Seq, Seq), device=q.device)
            mask = torch.triu(mask, diagonal=1)

            # (N, H, Seq_q, Seq_k)
            qk_T_normalized_masked = (qk_T_normalized * (1 - mask)) + (2e9 * mask)
            qk_T_normalized_masked[qk_T_normalized_masked>=2e9] = -torch.inf

            qk_T_softmax = F.softmax(qk_T_normalized_masked, dim=3)  # (N, H, Seq_q, Seq_k)
        else:
            qk_T_softmax = F.softmax(
                qk_T_normalized,
                dim=3)  # (N, H, Seq_q, Seq_k)

        # (N, H, Seq, D_split)
        attention_out = torch.einsum(
            "nhqk,nhkd->nhqd",
            qk_T_softmax,
            v_head_split)

        # Merge multi-head computations.
        N, Head, Seq, Dsplit = attention_out.shape
        attention_out = attention_out.permute(0, 2, 1, 3).reshape(
            N_q, Seq_q, Head*Dsplit)  # (N, Seq, D)
        return attention_out


# Self-Attention Block: (Attn -> Norm -> Residual).
class SelfAttentionBlock(nn.Module):
    def __init__(
            self,
            heads=8,
            in_dim=512,
            hidden_dim=512,
            use_masked_attn=True,
            activation_type="silu"):
        super().__init__()

        # Multi-Head Self Attention Layer.
        self.self_attn = AttentionLayer(
            heads=heads,
            in_dim=in_dim,
            hidden_dim=hidden_dim,
            use_cross_attn=False,
            use_masked_attn=use_masked_attn,
            activation_type=activation_type)
        self.self_attn_norm = nn.LayerNorm(
            in_dim)
        self.self_attn_res = ResidualLinearLayer(
            in_dim=in_dim,
            out_dim=in_dim,
            skip_dim=in_dim,
            activation_type=activation_type)

    def forward(self, x):
        init_x = x  # (N, Seq, D)

        x = self.self_attn(x)
        x = self.self_attn_norm(x)
        x = self.self_attn_res(
            x=x,
            x_skip=init_x)
        return x


# Cross-Attention Block: (Attn -> Norm -> Residual).
class CrossAttentionBlock(nn.Module):
    def __init__(
            self,
            heads=8,
            in_dim=512,
            cond_dim=512,
            hidden_dim=512,
            activation_type="silu"):
        super().__init__()

        # Multi-Head Cross Attention Layer.
        self.cross_attn = AttentionLayer(
            heads=heads,
            in_dim=in_dim,
            cond_dim=cond_dim,
            hidden_dim=hidden_dim,
            use_cross_attn=True,
            use_masked_attn=False,
            activation_type=activation_type)
        self.cross_attn_norm = nn.LayerNorm(
            in_dim)
        self.cross_attn_res = ResidualLinearLayer(
            in_dim=in_dim,
            out_dim=in_dim,
            skip_dim=in_dim,
            activation_type=activation_type)

    def forward(self, x, cross_cond):
        init_x = x  # (N, Seq, D)

        x = self.cross_attn(
            x=x,
            cross_cond=cross_cond)
        x = self.cross_attn_norm(x)
        x = self.cross_attn_res(
            x=x,
            x_skip=init_x)
        return x


# Transformer Block: https://arxiv.org/abs/1706.03762
class TransformerBlock(nn.Module):
    def __init__(
            self,
            in_dim=512,
            cond_dim=512,
            hidden_dim=512,
            self_attn_heads=8,
            cross_attn_heads=8,
            use_cross_attn=True,
            use_masked_attn=True,
            activation_type="silu"):
        super().__init__()

        self.use_cross_attn = use_cross_attn

        self.self_attn_block = SelfAttentionBlock(
            heads=self_attn_heads,
            in_dim=in_dim,
            hidden_dim=hidden_dim,
            use_masked_attn=use_masked_attn,
            activation_type=activation_type)
        if self.use_cross_attn:
            self.cross_attn_block = CrossAttentionBlock(
                heads=cross_attn_heads,
                in_dim=in_dim,
                cond_dim=cond_dim,
                hidden_dim=hidden_dim,
                activation_type=activation_type)
        self.feedforward_block = FeedforwardBlock(
            in_dim=in_dim,
            hidden_dim=hidden_dim,
            activation_type=activation_type)

    def forward(self, x, cross_cond=None):
        x = self.self_attn_block(x)
        if self.use_cross_attn:
            x = self.cross_attn_block(
                x,
                cross_cond=cross_cond)
        x = self.feedforward_block(x)

        return x
