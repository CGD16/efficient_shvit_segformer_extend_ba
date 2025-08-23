# from: https://github.com/ucasligang/SimViT/blob/main/classification/simvit.py
# converted to 3D

import math

import torch
import torch.nn as nn
from torch.nn.init import trunc_normal_

import unfoldNd


class CenterAttention3D(nn.Module):
    """
    CenterAttention3D class implements center attention for 3D input data.
    
    Args:
        dim (int): Dimension of the input data.
        num_heads (int, default=1): Number of attention heads.
        qkv_bias (boolean, default=True): If True, adds a learnable bias to query, key, and value projections.
        qk_scale (float or None, default=None): Scaling factor for query and key. If None, it is set to (dim // num_heads) ** -0.5.
        attn_drop (float, default=0.0): Dropout rate for attention weights.
        proj_drop (float, default=0.0): Dropout rate for output projection.
        stride (int, default=1): Stride of the attention.
        padding (boolean, default=True): If True, applies padding to the input.
        kernel_size (int, default=3): Size of the convolutional kernel used in attention.
       
    Inputs:
        x (torch.Tensor): A input tensor with shape (B, N, C), where B is the batch size, N is the number of tokens (or spatial size), and C is the number of channels.
        D (int): Depth of the input volume.
        H (int): Height of the input volume.
        W (int): Width of the input volume.
 
    Outputs:
        out (torch.Tensor): A output tensor with shape (B, N_out, C), where B is the batch size, N_out is the number of output tokens, and C is the number of channels.
    """    
    def __init__(self, dim: int, num_heads: int=1, qkv_bias: bool=True, qk_scale: float=None, attn_drop: float=0.,
                 proj_drop: float=0., stride: int=1, padding: bool=True, kernel_size: int=3):
        super(CenterAttention3D, self).__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."
        self.k_size = kernel_size  # kernel size
        self.stride = stride  # stride
 
        self.in_channels = dim  # origin channel is 3, patch channel is in_channel
        self.num_heads = num_heads
        self.head_channel = dim // num_heads
        self.pad_size = kernel_size // 2 if padding is True else 0  # padding size
        self.pad = nn.ZeroPad3d(self.pad_size)  # padding around the input
        self.scale = qk_scale or (dim // num_heads)**-0.5
        self.unfold = unfoldNd.UnfoldNd(kernel_size=self.k_size, stride=self.stride, padding=0, dilation=1)
 
        self.qkv_bias = qkv_bias
        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv_proj = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.softmax = nn.Softmax(dim=-1)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        
        self.apply(self._init_weights)


    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()        
        elif isinstance(m, nn.Conv3d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()        
 
 
    def forward(self, x: torch.Tensor, D: int, H: int, W: int) -> torch.Tensor:
        B, N, C = x.shape
        x = x.reshape(B, D, H, W, C)
        assert C == self.in_channels
 
        self.pat_size_d = (D+2 * self.pad_size-self.k_size) // self.stride+1
        self.pat_size_h = (H+2 * self.pad_size-self.k_size) // self.stride+1
        self.pat_size_w = (W+2 * self.pad_size-self.k_size) // self.stride+1
        self.num_patch = self.pat_size_d * self.pat_size_h * self.pat_size_w
 
        # (B, NumHeads, H, W, HeadC)
        q = self.q_proj(x).reshape(B, D, H, W, self.num_heads, self.head_channel).permute(0, 4, 1, 2, 3, 5)
        q = q.unsqueeze(dim=5)
        q = q * self.scale
        q = q.reshape(B, self.num_heads, self.num_patch, 1, self.head_channel)
 
        # (2, B, NumHeads, HeadsC, H, W)
        kv = self.kv_proj(x).reshape(B, D, H, W, 2, self.num_heads, self.head_channel).permute(4, 0, 5, 6, 1, 2, 3)
        kv = self.pad(kv)  # (2, B, NumH, HeadC, D, H, W)
        kv = kv.permute(0, 1, 2, 4, 5, 6, 3) # -> (2, B, NumH, D, H, W, HeadC)
        D, H, W = D + self.pad_size * 2, H + self.pad_size * 2, W + self.pad_size * 2
 
        # unfold plays role of conv3d to get patch data
        kv = kv.permute(0, 1, 2, 6, 3, 4, 5).reshape(2 * B, -1, D, H, W)
        kv = self.unfold(kv)
        kv = kv.reshape(2, B, self.num_heads, self.head_channel, self.k_size**3, self.num_patch)  # (2, B, NumH, HC, ks*ks*ks, NumPatch)
        kv = kv.permute(0, 1, 2, 5, 4, 3)  # (2, B, NumH, NumPatch, ks*ks*ks, HC)
        k, v = kv[0], kv[1]
 
        # (B, NumH, NumPatch, 1, HeadC)
        attn = (q @ k.transpose(-2, -1))  # (B, NumH, NumPatch, ks*ks*ks, ks*ks*ks)
        attn = self.softmax(attn)  # softmax last dim
        attn = self.attn_drop(attn)
 
        out = (attn @ v).squeeze(3)  # (B, NumH, NumPatch, HeadC)
        out = out.permute(0, 2, 1, 3).reshape(B, self.pat_size_d, self.pat_size_h, self.pat_size_w, C)  # (B, Pd, Ph, Pw, C)
        out = self.proj(out)
        out = self.proj_drop(out)
        out = out.reshape(B, -1, C)
 
        return out