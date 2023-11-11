# --------------------------------------------------------
# References:
# DiT: https://github.com/facebookresearch/DiT
# MAE: https://github.com/facebookresearch/mae/blob/main/models_mae.py
# --------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import math
from timm.models.vision_transformer import PatchEmbed, Mlp

# Postional Embedding
# https://github.com/facebookresearch/mae/blob/main/util/pos_embed.py

def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


class ClassEmbedding(nn.Module):
    def __init__(self, num_classes, hidden_size):
        super().__init__()
        self.embedding_table = nn.Embedding(num_classes, hidden_size)

    def forward(self, labels):
        return self.embedding_table(labels)


class DEQAttention(nn.Module):
    def __init__(
            self,
            dim,
            num_heads=8,
            qkv_bias=False,
            qk_norm=False,
            attn_drop=0.,
            proj_drop=0.,
            norm_layer=nn.LayerNorm,
            cond=False
    ):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.fast_attn = hasattr(torch.nn.functional, 'scaled_dot_product_attention')  # FIXME
        
        self.cond = cond

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, c=None):
        B, N, C = x.shape
        qkv = self.qkv(x)

        # Injection
        if self.cond:
            qkv = qkv + c

        qkv = qkv.reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        if self.fast_attn:
            x = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=self.attn_drop.p,
            )
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class ViTBlock(nn.Module):
    """
    A standard ViT block.
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, cond=False, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, eps=1e-6)
        self.attn = DEQAttention(hidden_size, num_heads=num_heads, qkv_bias=True, cond=cond, **block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)

        # For Pytorch 1.13
        act = lambda: nn.GELU()
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=act, drop=0)
 
        self.cond = cond

    def forward(self, x, c):
        if self.cond:
            x = x + self.attn(self.norm1(x), c)
            x = x + self.mlp(self.norm2(x))
        else:
            x = x + self.attn(self.norm1(x))
            x = x + self.mlp(self.norm2(x))
 
        return x


class FinalLayer(nn.Module):
    def __init__(self, hidden_size, patch_size, out_channels, cond=False):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        
    def forward(self, x):
        x = self.norm_final(x)
        x = self.linear(x)
        return x


class ViT(nn.Module):
    """
    Learning fast image generation using ViT.
    """
    def __init__(
        self,
        args,
        input_size=32,
        patch_size=2,
        in_channels=3,
        hidden_size=1152,
        depth=28,
        num_heads=16,
        mlp_ratio=4.0,
        num_classes=10,
        cond=False
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads
        
        self.cond = cond

        self.x_embedder = PatchEmbed(input_size, patch_size, in_channels, hidden_size, bias=True)
        
        if self.cond:
            self.y_embedder = ClassEmbedding(num_classes, 3*hidden_size)
        
        num_patches = self.x_embedder.num_patches
        # Will use fixed sin-cos embedding:
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_size), requires_grad=False)

        self.blocks = nn.ModuleList([
            ViTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio, cond=cond) for _ in range(depth)
        ])
        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels, cond=cond)
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            if isinstance(module, nn.LayerNorm):
                if hasattr(module, 'weight') and module.weight is not None:
                    nn.init.constant_(module.weight, 1)
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize (and freeze) pos_embed by sin-cos embedding:
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.x_embedder.num_patches ** 0.5))
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # Initialize patch_embed like nn.Linear:
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)

        # Initialize label embedding:
        if self.cond:
            nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)

        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def unpatchify(self, x):
        """
        x: (B, N, P ** 2 * C)
        imgs: (B, H, W, C)
        """
        c = self.out_channels
        p = self.x_embedder.patch_size[0]
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, h * p))
        return imgs

    def forward(self, x, y=None):
        """
        Forward pass of ViT.
        x: (B, C, H, W) tensor of spatial inputs (images or latent representations of images)
        y: (B,) tensor of class labels
        """
        x = self.x_embedder(x) + self.pos_embed     # (B, N, D), where N = H * W / P ** 2
        
        c = None
        if self.cond:
            c = self.y_embedder(y).view(B, 1, 3*C)

        for block in self.blocks:
            x = block(x, c)                         # (B, N, D)
        x = self.final_layer(x)                     # (B, N, P ** 2 * C_out)

        x = self.unpatchify(x)                      # (B, C_out, H, W)
        return x



#################################################################################
#                                   ViT Configs                                  #
#################################################################################

def ViT_XL_2(args, **kwargs):
    return ViT(args, depth=28, hidden_size=1152, patch_size=2, num_heads=16, **kwargs)

def ViT_XL_4(args, **kwargs):
    return ViT(args, depth=28, hidden_size=1152, patch_size=4, num_heads=16, **kwargs)

def ViT_XL_8(args, **kwargs):
    return ViT(args, depth=28, hidden_size=1152, patch_size=8, num_heads=16, **kwargs)

def ViT_L_2(args, **kwargs):
    return ViT(args, depth=24, hidden_size=1024, patch_size=2, num_heads=16, **kwargs)

def ViT_L_4(args, **kwargs):
    return ViT(args, depth=24, hidden_size=1024, patch_size=4, num_heads=16, **kwargs)

def ViT_L_8(args, **kwargs):
    return ViT(args, depth=24, hidden_size=1024, patch_size=8, num_heads=16, **kwargs)

def ViT_B_2(args, **kwargs):
    return ViT(args, depth=12, hidden_size=768, patch_size=2, num_heads=12, **kwargs)

def ViT_B_4(args, **kwargs):
    return ViT(args, depth=12, hidden_size=768, patch_size=4, num_heads=12, **kwargs)

def ViT_B_8(args, **kwargs):
    return ViT(args, depth=12, hidden_size=768, patch_size=8, num_heads=12, **kwargs)

def ViT_S_2(args, **kwargs):
    return ViT(args, depth=12, hidden_size=384, patch_size=2, num_heads=6, **kwargs)

def ViT_S_4(args, **kwargs):
    return ViT(args, depth=12, hidden_size=384, patch_size=4, num_heads=6, **kwargs)

def ViT_S_8(args, **kwargs):
    return ViT(args, depth=12, hidden_size=384, patch_size=8, num_heads=6, **kwargs)


ViT_models = {
    'ViT-XL/2': ViT_XL_2,  'ViT-XL/4': ViT_XL_4,  'ViT-XL/8': ViT_XL_8,
    'ViT-L/2':  ViT_L_2,   'ViT-L/4':  ViT_L_4,   'ViT-L/8':  ViT_L_8,
    'ViT-B/2':  ViT_B_2,   'ViT-B/4':  ViT_B_4,   'ViT-B/8':  ViT_B_8,
    'ViT-S/2':  ViT_S_2,   'ViT-S/4':  ViT_S_4,   'ViT-S/8':  ViT_S_8,
}
