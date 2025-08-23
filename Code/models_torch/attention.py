# from: https://github.com/IMvision12/SegFormer-tf/blob/main/models/Attention.py
# converted to pytorch

import torch
import torch.nn as nn
import math


class Attention(nn.Module):
    """
    2D Attention Layer.

    Args:
        dim (int): The number of input dimensions.
        num_heads (int): The number of attention heads.
        sr_ratio (int, optional): The spatial reduction ratio. Defaults to 1 (no reduction).
        qkv_bias (bool, optional): If True, adds a learnable bias to the query, key, and value projections. 
                                   Defaults to False.
        attn_drop (float, optional): Dropout rate on attention weights. Defaults to 0.0.
        proj_drop (float, optional): Dropout rate on the output projection. Defaults to 0.0.

    Inputs:
        x (torch.Tensor): A 3D input tensor of shape (B, N, C), where B is the batch size, 
                          N is the sequence length, and C is the number of input dimensions.
        H (int): The height of the input tensor.
        W (int): The width of the input tensor.

    Outputs:
        torch.Tensor: A 3D output tensor of shape (B, N, C) after applying the attention mechanism.
    """      
    def __init__(self, dim: int, num_heads: int, sr_ratio: int=1, qkv_bias: bool=False, attn_drop: float=0.0, proj_drop: float=0.0):
        super(Attention, self).__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = math.sqrt(self.head_dim)

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        B, N, C = x.shape

        q = self.q(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)  # (B, num_heads, N, head_dim)

        if self.sr_ratio > 1:
            x_reshaped = x.reshape(B, H, W, C).permute(0, 3, 1, 2)  # (B, C, H, W)
            x_reshaped = self.sr(x_reshaped).reshape(B, C, -1).permute(0, 2, 1)  # (B, N', C)
            x_reshaped = self.norm(x_reshaped)
        else:
            x_reshaped = x  # no downsampling necessary

        k = self.k(x_reshaped).reshape(B, -1, self.num_heads, self.head_dim).permute(0, 2, 3,
                                                                                     1)  # (B, num_heads, head_dim, N')
        v = self.v(x_reshaped).reshape(B, -1, self.num_heads, self.head_dim).permute(0, 2, 1,
                                                                                     3)  # (B, num_heads, N', head_dim)

        attn = (q @ k) / self.scale  # (B, num_heads, N, N')
        attn = torch.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).permute(0, 2, 1, 3).reshape(B, N, C)  # (B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x
