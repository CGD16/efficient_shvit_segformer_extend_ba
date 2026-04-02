# from: https://github.com/ucasligang/SimViT/blob/main/classification/simvit.py
# converted to 3D

import math

import torch
import torch.nn as nn
from torch.nn.init import trunc_normal_




class CenterAttention(nn.Module):
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
        H (int): Height of the input volume.
        W (int): Width of the input volume.
 
    Outputs:
        out (torch.Tensor): A output tensor with shape (B, N_out, C), where B is the batch size, N_out is the number of output tokens, and C is the number of channels.
    """    
    def __init__(self, dim, num_heads=1, qkv_bias=True, qk_scale=None, attn_drop=0.,
                 proj_drop=0., stride=1, padding=True, kernel_size=3):
        super(CenterAttention, self).__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."
        self.k_size = kernel_size  # kernel size
        self.stride = stride  # stride
 
        self.in_channels = dim  # origin channel is 3, patch channel is in_channel
        self.num_heads = num_heads
        self.head_channel = dim // num_heads
        self.pad_size = kernel_size // 2 if padding is True else 0  # padding size
        self.pad = nn.ZeroPad2d(self.pad_size)  # padding around the input
        self.scale = qk_scale or (dim // num_heads)**-0.5
        self.unfold = nn.Unfold(kernel_size=self.k_size, stride=self.stride, padding=0, dilation=1)
 
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
                
                
    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.reshape(B, H, W, C)
        assert C == self.in_channels
 
        self.pat_size_h = (H+2 * self.pad_size-self.k_size) // self.stride+1
        self.pat_size_w = (W+2 * self.pad_size-self.k_size) // self.stride+1
        self.num_patch = self.pat_size_h * self.pat_size_w
 
        # (B, NumHeads, H, W, HeadC)
        q = self.q_proj(x).reshape(B, H, W, self.num_heads, self.head_channel).permute(0, 3, 1, 2, 4)
        q = q.unsqueeze(dim=4)
        q = q * self.scale
        q = q.reshape(B, self.num_heads, self.num_patch, 1, self.head_channel)
 
        # (2, B, NumHeads, HeadsC, H, W)
        kv = self.kv_proj(x).reshape(B, H, W, 2, self.num_heads, self.head_channel).permute(3, 0, 4, 5, 1, 2)
        kv = self.pad(kv)  # (2, B, NumH, HeadC, H, W)
        kv = kv.permute(0, 1, 2, 4, 5, 3)
        H, W = H + self.pad_size * 2, W + self.pad_size * 2
 
        # unfold plays role of conv2d to get patch data
        kv = kv.permute(0, 1, 2, 5, 3, 4).reshape(2 * B, -1, H, W)
        kv = self.unfold(kv)
        kv = kv.reshape(2, B, self.num_heads, self.head_channel, self.k_size**2, self.num_patch)  # (2, B, NumH, HC, ks*ks, NumPatch)
        kv = kv.permute(0, 1, 2, 5, 4, 3)  # (2, B, NumH, NumPatch, ks*ks, HC)
        k, v = kv[0], kv[1]
 
        # (B, NumH, NumPatch, 1, HeadC)
        attn = (q @ k.transpose(-2, -1))  # (B, NumH, NumPatch, ks*ks, ks*ks)
        attn = self.softmax(attn)  # softmax last dim
        attn = self.attn_drop(attn)
 
        out = (attn @ v).squeeze(3)  # (B, NumH, NumPatch, HeadC)
        out = out.permute(0, 2, 1, 3).reshape(B, self.pat_size_h, self.pat_size_w, C)  # (B, Ph, Pw, C)
        out = self.proj(out)
        out = self.proj_drop(out)
        out = out.reshape(B, -1, C)
 
        return out