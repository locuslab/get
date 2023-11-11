# --------------------------------------------------------
# References:
# DiT: https://github.com/facebookresearch/DiT
# MAE: https://github.com/facebookresearch/mae
# --------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import math
from timm.models.vision_transformer import PatchEmbed, Attention, Mlp

from torchdeq import get_deq
from torchdeq.norm import apply_norm, reset_norm
from torchdeq.utils import mem_gc


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


class AttnInterface(nn.Module):
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

    def forward(self, x, c, u=None):
        B, N, C = x.shape
        qkv = self.qkv(x)
        
        # Injection
        if self.cond:
            qkv = qkv + c
        if u is not None:
            qkv = qkv + u

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


class GETBlock(nn.Module):
    """
    A GET block with additive attention injection.
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, cond=False, **block_kwargs):
        super().__init__()
        # Attention
        self.norm1 = nn.LayerNorm(hidden_size, eps=1e-6)
        self.attn = AttnInterface(hidden_size, num_heads=num_heads, qkv_bias=True, cond=cond, **block_kwargs)
        
        # MLP
        self.norm2 = nn.LayerNorm(hidden_size, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)

        act = lambda: nn.GELU()
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=act, drop=0)
 
    def forward(self, x, c, u=None):
        x = x + self.attn(self.norm1(x), c, u)
        x = x + self.mlp(self.norm2(x))
        return x


class FinalLayer(nn.Module):
    """
    The final projection layer.
    """
    def __init__(self, hidden_size, patch_size, out_channels, cond=False):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        
    def forward(self, x):
        x = self.norm_final(x)
        x = self.linear(x)
        return x


class GET(nn.Module):
    """
    Diffusion model with a Transformer backbone.
    """
    def __init__(
        self,
        args,
        input_size=32,
        patch_size=2,
        in_channels=3,
        hidden_size=1152,
        depth=28,
        deq_depth=3,
        num_heads=16,
        mlp_ratio=4.0,
        deq_mlp_ratio=16.0,
        num_classes=10,
        cond=False
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.deq_depth = deq_depth

        self.cond = cond

        self.x_embedder = PatchEmbed(input_size, patch_size, in_channels, hidden_size, bias=True)
        
        if self.cond:
            self.y_embedder = ClassEmbedding(num_classes, 3*hidden_size)
        
        num_patches = self.x_embedder.num_patches
        # Will use fixed sin-cos embedding:
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_size), requires_grad=False)

        self.blocks = nn.ModuleList([
            GETBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio, cond=cond) for _ in range(depth)
        ])

        # injection
        self.qkv_inj = nn.Linear(hidden_size, hidden_size*3*deq_depth, bias=False)

        # DEQ blocks
        self.deq_blocks = nn.ModuleList([
            GETBlock(hidden_size, num_heads, mlp_ratio=deq_mlp_ratio, cond=cond) for _ in range(deq_depth)
        ])
        
        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels, cond=cond)
        self.initialize_weights()
        
        self.mem = args.mem
        self.deq = get_deq(args)
        apply_norm(self.deq_blocks, args=args)

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            if isinstance(module, nn.LayerNorm):
                if module.weight is not None:
                    nn.init.constant_(module.weight, 1)
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize (and freeze) pos_embed by sin-cos embedding:
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.x_embedder.num_patches ** 0.5))
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # Initialize patch embedding:
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)

        # Initialize class embedding table:
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
    
    def decode(self, z):
        x = self.final_layer(z)                     # (B, N, P ** 2 * C_out)
        x = self.unpatchify(x)                      # (B, C_out, H, W)
        return x

    def forward(self, x, y=None):
        """
        Forward pass of GET.
        x: (B, C, H, W) tensor of spatial inputs (images or latent representations of images)
        y: (B,) tensor of class labels
        """
        reset_norm(self)

        x = self.x_embedder(x) + self.pos_embed     # (B, N, D), where N = H * W / P ** 2
        B, N, C = x.shape

        c = None
        if self.cond:
            c = self.y_embedder(y).view(B, 1, 3*C)
        
        # Injection T
        for block in self.blocks:
            x = block(x, c)                         # (B, N, D)

        u = self.qkv_inj(x)
        u_list = u.chunk(self.deq_depth, dim=-1)

        def func(z):
            for block, u in zip(self.deq_blocks, u_list):
                if self.mem:
                    z = mem_gc(block, (z, c, u))
                else:
                    z = block(z, c, u)
            return z
        
        # Equilibrium T
        z = torch.randn_like(x)
        z_out, info = self.deq(func, z)       
        
        if self.training:
            # For fixed point correction
            return [self.decode(z) for z in z_out]
        else:
            return self.decode(z_out[-1])


#################################################################################
#                         Current GET Configs                                   #
#################################################################################


def GET_T_2_L6_L3_H6(args, **kwargs):
    return GET(args, depth=6, hidden_size=256, patch_size=2, num_heads=4, deq_mlp_ratio=6, **kwargs)

def GET_M_2_L6_L3_H6(args, **kwargs):
    return GET(args, depth=6, hidden_size=384, patch_size=2, num_heads=6, deq_mlp_ratio=6, **kwargs)

def GET_S_2_L6_L3_H8(args, **kwargs):
    return GET(args, depth=6, hidden_size=512, patch_size=2, num_heads=8, deq_mlp_ratio=8, **kwargs)

def GET_B_2_L1_L3_H12(args, **kwargs):
    return GET(args, depth=1, hidden_size=768, patch_size=2, num_heads=12, deq_mlp_ratio=12, **kwargs)

def GET_B_2_L6_L3_H8(args, **kwargs):
    return GET(args, depth=6, hidden_size=768, patch_size=2, num_heads=12, deq_mlp_ratio=8, **kwargs)


GET_models = {
    'GET-T/2': GET_T_2_L6_L3_H6,
    'GET-M/2': GET_M_2_L6_L3_H6,
    'GET-S/2': GET_S_2_L6_L3_H8,
    'GET-B/2': GET_B_2_L1_L3_H12,
    'GET-B/2+': GET_B_2_L6_L3_H8,
}
